#include <Arduino.h>
#include <SPI.h>
#include "registers.h"

// ----------------- Your existing helpers -----------------
static inline void dvga_cs_low() { digitalWrite(DVGA_CS_PIN, LOW); }
static inline void dvga_cs_high() { digitalWrite(DVGA_CS_PIN, HIGH); }

static inline void dvgaI_cs_low() { digitalWrite(DVGAI_CS_PIN, LOW); }
static inline void dvgaI_cs_high() { digitalWrite(DVGAI_CS_PIN, HIGH); }

static inline void le_low() { digitalWrite(LE, LOW); }
static inline void le_high() { digitalWrite(LE, HIGH); }

void att_write_reg(att_reg_address addr, uint8_t data)
{
  SPI.beginTransaction(SPISettings(100000, LSBFIRST, SPI_MODE0));
  le_low();
  SPI.transfer(data);
  SPI.transfer(((uint8_t)addr) << 4);
  le_high();
  le_low();
  le_high();
  SPI.endTransaction();
}

void att_init_serial()
{
  digitalWrite(LE, HIGH); // latch enabled
  digitalWrite(PBAR, HIGH);
}

void att_serial(uint8_t g)
{
  g &= 0x7F; // keep only 7 bits (0..127)
  att_write_reg(att_reg_address::REG_ATTN, g);
}

void dvga_write_reg(uint8_t data)
{
  SPI.beginTransaction(SPISettings(100000, LSBFIRST, SPI_MODE0));
  dvga_cs_high();
  delay(10);

  SPI.transfer(data);
  SPI.transfer(0x00);
  dvga_cs_low();
  SPI.endTransaction();
}

void dvgaI_write_reg(uint8_t data)
{
  SPI.beginTransaction(SPISettings(100000, LSBFIRST, SPI_MODE0));
  dvgaI_cs_high();
  delay(10);

  SPI.transfer(data);
  SPI.transfer(0x00);
  dvgaI_cs_low();
  SPI.endTransaction();
}

void dvga_serial(uint8_t g)
{
  g &= 0x3F; // keep only 6 bits (0..63)
  g ^= 0x3F;
  dvga_write_reg(g);
  dvgaI_write_reg(g);
}

void demod_init()
{
  demod_set_default();
  demod_write_reg(demod_reg_address::ADDR_ATT_IP3IC, 0x04);
  demod_write_reg(demod_reg_address::ADDR_BAND_LF1_CF2, 0xAA);
  demod_write_reg(demod_reg_address::ADDR_LVCM_CF1, 0x42);
}

// ----------------- ADC sampling + streaming -----------------

// --- Pins ---
constexpr uint8_t CONVST_PIN = 2; // D2 = PB25 = TIOA0
constexpr uint8_t CS1_PIN = 10;   // SPI CS0 (hardware)
constexpr uint8_t CS2_PIN = 4;    // SPI CS1 (hardware)

// --- Timing targets ---
constexpr uint32_t CONVST_LOW_NS = 100;
constexpr uint32_t CONVST_HIGH_NS = 12400 * 5; // 12.5us period => 80kSPS
constexpr uint32_t TCONV_MAX_NS = 8800;        // ADS8867 tconv max

// SPI clock
constexpr uint32_t SPI_SCK_HZ = 14000000;

// --------- Ring buffer for samples ---------
struct SamplePair
{
  uint16_t a1;
  uint16_t a2;
};

// Try RB_BITS=14 if you have RAM margin (64KB). If unstable, revert to 13.
constexpr uint32_t RB_BITS = 14; // 2^14 = 16384 pairs
constexpr uint32_t RB_SIZE = (1u << RB_BITS);
constexpr uint32_t RB_MASK = RB_SIZE - 1;

SamplePair rb[RB_SIZE];
volatile uint32_t rb_head = 0;
volatile uint32_t rb_tail = 0;
volatile uint32_t rb_overflow = 0;
volatile uint32_t seq = 0;

// Sampling gate (so we don’t overflow before PC reads)
volatile bool sampling_enabled = false;

// ---------- Helpers ----------
static inline uint32_t ns_to_ticks(uint32_t ns, uint32_t clk_hz)
{
  return (uint32_t)(((uint64_t)ns * clk_hz + 500000000ULL) / 1000000000ULL);
}

// PCS codes for variable select mode
static constexpr uint8_t PCS_NPCS0 = 0; // D10
static constexpr uint8_t PCS_NPCS1 = 1; // D4

static inline uint16_t spi0_read16(uint8_t pcs_code)
{
  while (!(SPI0->SPI_SR & SPI_SR_TDRE))
  {
  }
  SPI0->SPI_TDR = (uint32_t)(0x0000u) | ((uint32_t)(pcs_code & 0xF) << 16);
  while (!(SPI0->SPI_SR & SPI_SR_RDRF))
  {
  }
  return (uint16_t)(SPI0->SPI_RDR & 0xFFFFu);
}

static void setup_spi_hw_cs()
{
  SPI.begin();

  pmc_enable_periph_clk(ID_SPI0);
  pmc_enable_periph_clk(ID_PIOA);

  // Hand PA28/PA29 (NPCS0/NPCS1) to peripheral A
  const uint32_t PA28 = (1u << 28);
  const uint32_t PA29 = (1u << 29);
  PIOA->PIO_PDR = PA28 | PA29;
  PIOA->PIO_ABSR &= ~(PA28 | PA29);
  PIOA->PIO_PUDR = PA28 | PA29;

  SPI0->SPI_CR = SPI_CR_SWRST;
  SPI0->SPI_CR = SPI_CR_SWRST;

  SPI0->SPI_MR = SPI_MR_MSTR | SPI_MR_MODFDIS | SPI_MR_PS | SPI_MR_DLYBCS(6);

  uint8_t scbr = (uint8_t)(VARIANT_MCK / SPI_SCK_HZ);
  if (scbr < 1)
    scbr = 1;
  if (scbr > 255)
    scbr = 255;

  const uint32_t csr_common =
      SPI_CSR_SCBR(scbr) |
      SPI_CSR_BITS_16_BIT |
      SPI_CSR_CSNAAT |
      SPI_CSR_NCPHA; // CPOL=0

  SPI0->SPI_CSR[0] = csr_common; // CS0 (D10)
  SPI0->SPI_CSR[1] = csr_common; // CS1 (D4)

  SPI0->SPI_CR = SPI_CR_SPIEN;
  (void)SPI0->SPI_RDR;
}

static void setup_convst_and_timer_trigger_config_only()
{
  const uint32_t tc_clk_hz = VARIANT_MCK / 2; // 42 MHz

  uint32_t low_ticks = ns_to_ticks(CONVST_LOW_NS, tc_clk_hz);
  uint32_t high_ticks = ns_to_ticks(CONVST_HIGH_NS, tc_clk_hz);
  if (low_ticks < 2)
    low_ticks = 2;
  if (high_ticks < 2)
    high_ticks = 2;
  const uint32_t period_ticks = low_ticks + high_ticks;

  uint32_t tconv_ticks = ns_to_ticks(TCONV_MAX_NS, tc_clk_hz);
  if (tconv_ticks < 1)
    tconv_ticks = 1;

  const uint32_t read_tick = low_ticks + tconv_ticks;

  pmc_enable_periph_clk(ID_TC0);
  pmc_enable_periph_clk(ID_TC1);

  // Route D2 (PB25) to TIOA0 (Peripheral B)
  pmc_enable_periph_clk(ID_PIOB);
  PIOB->PIO_PDR |= PIO_PDR_P25;
  PIOB->PIO_ABSR |= PIO_ABSR_P25;

  // TC0 ch0: waveform on TIOA0 (CONVST)
  TC0->TC_CHANNEL[0].TC_CCR = TC_CCR_CLKDIS;
  TC0->TC_CHANNEL[0].TC_IDR = 0xFFFFFFFF;
  TC0->TC_CHANNEL[0].TC_CMR =
      TC_CMR_TCCLKS_TIMER_CLOCK1 |
      TC_CMR_WAVE |
      TC_CMR_WAVSEL_UP_RC |
      TC_CMR_ACPA_SET |
      TC_CMR_ACPC_CLEAR;
  TC0->TC_CHANNEL[0].TC_RA = low_ticks;
  TC0->TC_CHANNEL[0].TC_RC = period_ticks;

  // TC0 ch1: compare interrupt at read_tick each cycle
  TC0->TC_CHANNEL[1].TC_CCR = TC_CCR_CLKDIS;
  TC0->TC_CHANNEL[1].TC_IDR = 0xFFFFFFFF;
  TC0->TC_CHANNEL[1].TC_CMR =
      TC_CMR_TCCLKS_TIMER_CLOCK1 |
      TC_CMR_WAVE |
      TC_CMR_WAVSEL_UP_RC;
  TC0->TC_CHANNEL[1].TC_RA = read_tick;
  TC0->TC_CHANNEL[1].TC_RC = period_ticks;
  TC0->TC_CHANNEL[1].TC_IER = TC_IER_CPAS;

  // IMPORTANT: don’t starve USB interrupts (0 was too aggressive)
  NVIC_SetPriority(TC1_IRQn, 1);
  NVIC_EnableIRQ(TC1_IRQn);

  // Do NOT enable clocks here; we start when PC is ready
}

static void start_sampling()
{
  rb_head = rb_tail = 0;
  rb_overflow = 0;
  seq = 0;

  sampling_enabled = true;

  TC0->TC_CHANNEL[0].TC_CCR = TC_CCR_CLKEN;
  TC0->TC_CHANNEL[1].TC_CCR = TC_CCR_CLKEN;
  TC0->TC_BCR = TC_BCR_SYNC;
}

static void stop_sampling()
{
  sampling_enabled = false;
  TC0->TC_CHANNEL[0].TC_CCR = TC_CCR_CLKDIS;
  TC0->TC_CHANNEL[1].TC_CCR = TC_CCR_CLKDIS;
}

// ISR: acquire and push into ring buffer (NO serial here!)
void TC1_Handler()
{
  uint32_t sr = TC0->TC_CHANNEL[1].TC_SR;
  if ((sr & TC_SR_CPAS) && sampling_enabled)
  {
    SamplePair s;
    s.a1 = spi0_read16(PCS_NPCS0);
    s.a2 = spi0_read16(PCS_NPCS1);

    uint32_t head = rb_head;
    uint32_t next = (head + 1) & RB_MASK;

    if (next == rb_tail)
    {
      rb_overflow++;
    }
    else
    {
      rb[head] = s;
      __DMB();
      rb_head = next;
    }
    seq++;
  }
}

// ---------- USB packet format ----------
// Header: 'A''D''C''\n' (4 bytes)
// uint32_t seq_snapshot (4 bytes)
// uint16_t count        (2 bytes)
// Payload: count * SamplePair (count*4 bytes)
// Total header = 10 bytes
static void send_samples_usb()
{
  if (!SerialUSB)
    return;

  // BIG blocks = far fewer packets/writes than 12
  constexpr uint16_t MAX_SAMPLES_PER_BLOCK = 256; // 256*4=1024 bytes payload

  // Drain multiple blocks per loop() to catch up
  for (int blocks = 0; blocks < 8; blocks++)
  {
    uint32_t tail = rb_tail;
    uint32_t head = rb_head;
    if (tail == head)
      return;

    uint32_t available = (head - tail) & RB_MASK;
    uint16_t n = (available > MAX_SAMPLES_PER_BLOCK) ? MAX_SAMPLES_PER_BLOCK : (uint16_t)available;

    // Header
    uint8_t hdr[10];
    hdr[0] = 'A';
    hdr[1] = 'D';
    hdr[2] = 'C';
    hdr[3] = '\n';
    uint32_t seq_snap = seq;
    memcpy(&hdr[4], &seq_snap, 4);
    memcpy(&hdr[8], &n, 2);

    SerialUSB.write(hdr, 10);

    // Payload with wrap handling
    uint32_t to_end = RB_SIZE - tail;
    uint16_t n1 = (n > to_end) ? (uint16_t)to_end : n;
    uint16_t n2 = n - n1;

    SerialUSB.write((const uint8_t *)&rb[tail], n1 * sizeof(SamplePair));
    if (n2)
      SerialUSB.write((const uint8_t *)&rb[0], n2 * sizeof(SamplePair));

    rb_tail = (tail + n) & RB_MASK;
  }
}

// ---------- Test Mode State Machine ----------
enum TestState
{
  STATE_ADC_SAMPLING,
  STATE_WAIT_AFTER_ADC,
  STATE_DVGA_WRITING
};

volatile TestState currentState = STATE_ADC_SAMPLING;
volatile uint32_t stateStartTime = 0;
volatile uint8_t dvgaGainValue = 0; // Cycles 0-63 for DVGA test

void setup()
{
  pinMode(LE, OUTPUT);
  pinMode(PBAR, OUTPUT);
  digitalWrite(LE, HIGH);

  pinMode(DEMOD_CS_PIN, OUTPUT);
  digitalWrite(DEMOD_CS_PIN, HIGH);

  pinMode(DVGA_CS_PIN, OUTPUT);
  pinMode(OE, OUTPUT);
  pinMode(DVGAI_CS_PIN, OUTPUT);
  pinMode(DVGAI_OE, OUTPUT);

  digitalWrite(DVGA_CS_PIN, LOW);
  digitalWrite(OE, HIGH);   // enable output
  digitalWrite(SENB, HIGH); // enable serial mode

  digitalWrite(DVGAI_CS_PIN, LOW);
  digitalWrite(DVGAI_OE, HIGH); // enable output
  digitalWrite(SENB, HIGH);

  SPI.begin();

  demod_init();
  att_init_serial();
  att_serial((uint8_t)1);
  dvga_serial((uint8_t)32);

  // Native USB port (baud ignored)
  SerialUSB.begin(115200);

  setup_spi_hw_cs();
  setup_convst_and_timer_trigger_config_only();

  // Wait for PC to open the port (prevents initial overflow)
  // If you prefer auto-run without waiting, remove this block and call start_sampling() directly.
  uint32_t t0 = millis();
  while (!SerialUSB && (millis() - t0 < 5000))
  { /* wait up to 5s */
  }

  // Start sampling immediately once USB is up
  if (SerialUSB)
  {
    stateStartTime = millis(); // Initialize state timer
    start_sampling();
    SerialUSB.println("=== Test Mode Started ===");
    SerialUSB.println("Cycle: ADC(2s) -> Wait(1s) -> DVGA(2s) -> Repeat");
  }
}

// Re-initialize SPI for software mode (DVGA writes)
static void reinit_spi_software_mode()
{
  // Disable hardware SPI
  SPI0->SPI_CR = SPI_CR_SPIDIS;

  // Re-init Arduino SPI library for software-style use
  SPI.end();
  SPI.begin();
}

// Re-initialize SPI for hardware ADC mode
static void reinit_spi_hw_adc_mode()
{
  setup_spi_hw_cs();
}

void loop()
{
  uint32_t now = millis();

  // Optional: allow host to control start/stop via serial
  while (SerialUSB && SerialUSB.available())
  {
    int c = SerialUSB.read();
    if (c == 'S')
    {
      // Reset to ADC sampling state
      currentState = STATE_ADC_SAMPLING;
      stateStartTime = now;
      reinit_spi_hw_adc_mode();
      start_sampling();
    }
    else if (c == 'P')
    {
      stop_sampling();
    }
  }

  switch (currentState)
  {
  case STATE_ADC_SAMPLING:
    // Continue sending samples while in this state
    send_samples_usb();

    // After 2 seconds, transition to wait state
    if (now - stateStartTime >= 2000)
    {
      stop_sampling();

      // Print status
      if (SerialUSB)
      {
        SerialUSB.println("\n--- ADC sampling stopped (2s done) ---");
        SerialUSB.print("Samples collected, overflow count: ");
        SerialUSB.println(rb_overflow);
      }

      currentState = STATE_WAIT_AFTER_ADC;
      stateStartTime = now;
    }
    break;

  case STATE_WAIT_AFTER_ADC:
    // Wait for 1 second
    if (now - stateStartTime >= 1000)
    {
      if (SerialUSB)
      {
        SerialUSB.println("--- Starting DVGA write phase (2s) ---");
      }

      // Reconfigure SPI for DVGA (software mode)
      reinit_spi_software_mode();

      currentState = STATE_DVGA_WRITING;
      stateStartTime = now;
    }
    break;

  case STATE_DVGA_WRITING:
    // Write to DVGA continuously for 2 seconds
    // Change gain value periodically to demonstrate activity
    {
      static uint32_t lastDvgaWrite = 0;
      if (now - lastDvgaWrite >= 100) // Write every 100ms
      {
        dvga_serial(dvgaGainValue);

        if (SerialUSB)
        {
          SerialUSB.print("DVGA gain set to: ");
          SerialUSB.println(dvgaGainValue);
        }

        // Cycle through gain values 0-63
        dvgaGainValue = (dvgaGainValue + 4) & 0x3F;
        lastDvgaWrite = now;
      }
    }

    // After 2 seconds, go back to ADC sampling
    if (now - stateStartTime >= 2000)
    {
      if (SerialUSB)
      {
        SerialUSB.println("--- DVGA write phase done, restarting ADC ---");
      }

      // Reconfigure SPI for ADC hardware mode
      reinit_spi_hw_adc_mode();

      // Start sampling again
      start_sampling();

      currentState = STATE_ADC_SAMPLING;
      stateStartTime = now;
    }
    break;
  }
}
