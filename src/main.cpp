#include <Arduino.h>
#include <SPI.h>
#include "registers.h"

// ----------------- Attenuator Mode Selection -----------------
// Set to true for parallel mode, false for serial mode
bool attParallelMode = true;

// Parallel mode pin array
const uint8_t attParallelPins[7] = {ATT_D0, ATT_D1, ATT_D2, ATT_D3, ATT_D4, ATT_D5, ATT_D6};

// ----------------- DVGA Mode Selection -----------------
// Set to true for parallel mode, false for serial mode
bool dvgaParallelMode = true;

// DVGA parallel mode pin arrays
const uint8_t dvgaParallelPins[6] = {DVGA_D0, DVGA_D1, DVGA_D2, DVGA_D3, DVGA_D4, DVGA_D5};
const uint8_t dvgaIParallelPins[6] = {DVGAI_D0, DVGAI_D1, DVGAI_D2, DVGAI_D3, DVGAI_D4, DVGAI_D5};

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
  delayMicroseconds(10);
  SPI.transfer(data);
  SPI.transfer(((uint8_t)addr) << 4);
  delayMicroseconds(10);
  le_high();
  delayMicroseconds(10);
  le_low();
  delayMicroseconds(10);
  le_high();
  delayMicroseconds(10);
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

// Attenuator parallel mode functions
void att_init_parallel()
{
  // Set parallel pins as outputs
  for (uint8_t i = 0; i < 7; i++)
  {
    pinMode(attParallelPins[i], OUTPUT);
    digitalWrite(attParallelPins[i], LOW);
  }
  digitalWrite(PBAR, LOW); // Enable parallel mode
  digitalWrite(LE, HIGH);  // Latch enabled
}

void att_parallel(uint8_t g)
{
  g &= 0x7F; // keep only 7 bits (0..127)

  // Set each bit on the parallel pins
  for (uint8_t i = 0; i < 7; i++)
  {
    digitalWrite(attParallelPins[i], ((g >> i) & 0x01) ? HIGH : LOW);
  }

  // Pulse LE to latch the data
  le_low();
  delayMicroseconds(10);
  le_high();
  delayMicroseconds(10);
}

// Unified attenuator set function - uses current mode
void att_set(uint8_t g)
{
  if (attParallelMode)
  {
    att_parallel(g);
  }
  else
  {
    att_serial(g);
  }
}

// Switch attenuator mode
void att_set_mode(bool parallel)
{
  attParallelMode = parallel;
  if (parallel)
  {
    att_init_parallel();
  }
  else
  {
    att_init_serial();
  }
}

void dvga_write_reg(uint8_t data)
{
  SPI.beginTransaction(SPISettings(100000, LSBFIRST, SPI_MODE0));
  dvga_cs_high(); // Select chip (CS active HIGH)
  delayMicroseconds(100);

  SPI.transfer(data);
  delayMicroseconds(10);
  SPI.transfer(data);

  delayMicroseconds(100);
  dvga_cs_low(); // Deselect chip to latch data
  delayMicroseconds(10);
  SPI.endTransaction();
}

void dvgaI_write_reg(uint8_t data)
{
  SPI.beginTransaction(SPISettings(100000, LSBFIRST, SPI_MODE0));
  dvgaI_cs_high(); // Select chip (CS active HIGH)
  delayMicroseconds(100);

  SPI.transfer(data);
  delayMicroseconds(10);
  SPI.transfer(data);

  delayMicroseconds(100);
  dvgaI_cs_low(); // Deselect chip to latch data
  delayMicroseconds(10);
  SPI.endTransaction();
}

// DVGA serial mode write (writes to both ICs)
void dvga_serial(uint8_t g)
{
  g &= 0x3F; // keep only 6 bits (0..63)
  dvga_write_reg(g);
  delayMicroseconds(10);
  dvgaI_write_reg(g);
}

// DVGA serial mode initialization
void dvga_init_serial()
{
  digitalWrite(SENB, HIGH);       // Enable serial mode for DVGA1
  digitalWrite(DVGAI_SENB, HIGH); // Enable serial mode for DVGA2
  digitalWrite(DVGA_CS_PIN, LOW);
  digitalWrite(DVGAI_CS_PIN, LOW);
}

// DVGA parallel mode initialization
void dvga_init_parallel()
{
  // Set DVGA1 parallel pins as outputs
  for (uint8_t i = 0; i < 6; i++)
  {
    pinMode(dvgaParallelPins[i], OUTPUT);
    digitalWrite(dvgaParallelPins[i], LOW);
  }

  // Set DVGA2 parallel pins as outputs
  for (uint8_t i = 0; i < 6; i++)
  {
    pinMode(dvgaIParallelPins[i], OUTPUT);
    digitalWrite(dvgaIParallelPins[i], LOW);
  }

  digitalWrite(SENB, LOW);       // Enable parallel mode for DVGA1
  digitalWrite(DVGAI_SENB, LOW); // Enable parallel mode for DVGA2
}

// DVGA parallel mode write (writes to both ICs)
void dvga_parallel(uint8_t g)
{
  g &= 0x3F; // keep only 6 bits (0..63)

  // Write to DVGA1
  for (uint8_t i = 0; i < 6; i++)
  {
    digitalWrite(dvgaParallelPins[i], ((g >> i) & 0x01) ? HIGH : LOW);
  }

  // Write to DVGA2
  for (uint8_t i = 0; i < 6; i++)
  {
    digitalWrite(dvgaIParallelPins[i], ((g >> i) & 0x01) ? HIGH : LOW);
  }

  delayMicroseconds(10);
}

// Unified DVGA set function - uses current mode
void dvga_set(uint8_t g)
{
  if (dvgaParallelMode)
  {
    dvga_parallel(g);
  }
  else
  {
    dvga_serial(g);
  }
}

// Switch DVGA mode
void dvga_set_mode(bool parallel)
{
  dvgaParallelMode = parallel;
  if (parallel)
  {
    dvga_init_parallel();
  }
  else
  {
    dvga_init_serial();
  }
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

// --- Timing targets (16 kHz sample rate) ---
constexpr uint32_t CONVST_LOW_NS = 100;
constexpr uint32_t CONVST_HIGH_NS = 62400; // 62.5us total period => 16kSPS
constexpr uint32_t TCONV_MAX_NS = 8800;    // ADS8867 tconv max

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
  const uint32_t tc_clk_hz = VARIANT_MCK / 2; // 42 MHz (TIMER_CLOCK1 = MCK/2)

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
  STATE_DVGA_WRITING,
  STATE_WAIT_FOR_GAIN,
  STATE_DIRECT_DVGA_WRITE,
  STATE_DIRECT_ATT_WRITE
};

volatile TestState currentState = STATE_ADC_SAMPLING;
volatile uint32_t stateStartTime = 0;
volatile uint8_t dvgaGainValue = 0;       // Cycles 0-63 for DVGA test
volatile uint8_t receivedGainValue = 0;   // Gain value received from Python
volatile bool gainReceived = false;       // Flag to indicate gain was received
volatile uint8_t receivedAttValue = 0;    // Attenuator value received from Python
volatile bool attReceived = false;        // Flag to indicate attenuator value was received
volatile bool samplingWasRunning = false; // Track if sampling was running before direct write

void setup()
{
  pinMode(LE, OUTPUT);
  pinMode(PBAR, OUTPUT);
  digitalWrite(LE, HIGH);

  pinMode(DEMOD_CS_PIN, OUTPUT);
  digitalWrite(DEMOD_CS_PIN, HIGH);

  pinMode(DVGA_CS_PIN, OUTPUT);
  pinMode(OE, OUTPUT);
  pinMode(SENB, OUTPUT);
  pinMode(DVGA_DENA, OUTPUT);
  pinMode(DVGA_DENB, OUTPUT);
  pinMode(DVGAI_CS_PIN, OUTPUT);
  pinMode(DVGAI_OE, OUTPUT);
  pinMode(DVGAI_SENB, OUTPUT);
  pinMode(DVGAI_DENA, OUTPUT);
  pinMode(DVGAI_DENB, OUTPUT);

  digitalWrite(DVGA_CS_PIN, LOW);
  digitalWrite(OE, HIGH);        // enable output
  digitalWrite(DVGA_DENA, HIGH); // enable channel A
  digitalWrite(DVGA_DENB, HIGH); // enable channel B
  digitalWrite(DVGAI_CS_PIN, LOW);
  digitalWrite(DVGAI_OE, HIGH);   // enable output
  digitalWrite(DVGAI_DENA, HIGH); // enable channel A
  digitalWrite(DVGAI_DENB, HIGH); // enable channel B

  // Initialize DVGA based on mode
  if (dvgaParallelMode)
  {
    dvga_init_parallel();
  }
  else
  {
    dvga_init_serial();
  }

  SPI.begin();

  demod_init();

  // Initialize attenuator based on mode
  if (attParallelMode)
  {
    att_init_parallel();
  }
  else
  {
    att_init_serial();
  }
  att_set((uint8_t)1);
  // dvga_serial((uint8_t)32);

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
    else if (c == 'G')
    {
      stop_sampling();
      currentState = STATE_WAIT_AFTER_ADC;
    }
    else if (c == 'V')
    {
      // Receive gain value: format "V<value>" where value is 0-63
      // Wait for the value bytes
      uint32_t timeout = millis() + 100;
      while (!SerialUSB.available() && millis() < timeout)
      {
      }
      if (SerialUSB.available())
      {
        receivedGainValue = (uint8_t)SerialUSB.parseInt();
        receivedGainValue &= 0x3F; // Clamp to 0-63
        gainReceived = true;
        // Direct write to DVGA via state machine
        samplingWasRunning = sampling_enabled;
        if (samplingWasRunning)
        {
          stop_sampling();
        }
        currentState = STATE_WAIT_AFTER_ADC;
        stateStartTime = now;
      }
    }
    else if (c == 'A')
    {
      // Receive attenuator value: format "A<value>" where value is 0-127
      uint32_t timeout = millis() + 100;
      while (!SerialUSB.available() && millis() < timeout)
      {
      }
      if (SerialUSB.available())
      {
        receivedAttValue = (uint8_t)SerialUSB.parseInt();
        receivedAttValue &= 0x7F; // Clamp to 0-127
        attReceived = true;
        // Direct write to attenuator via state machine
        samplingWasRunning = sampling_enabled;
        if (samplingWasRunning)
        {
          stop_sampling();
        }
        currentState = STATE_DIRECT_ATT_WRITE;
        stateStartTime = now;
      }
    }
    else if (c == 'D')
    {
      // Demodulator register write: format "D<reg_id><value>"
      // reg_id: 0=DCOI, 1=DCOQ, 2=HD2IX, 3=HD2IY, 4=HD2QX, 5=HD2QY,
      //         6=HD3IX, 7=HD3IY, 8=HD3QX, 9=HD3QY
      uint32_t timeout = millis() + 100;
      while (!SerialUSB.available() && millis() < timeout)
      {
      }
      if (SerialUSB.available())
      {
        char regId = SerialUSB.read();
        timeout = millis() + 100;
        while (!SerialUSB.available() && millis() < timeout)
        {
        }
        if (SerialUSB.available())
        {
          uint8_t value = (uint8_t)SerialUSB.parseInt();
          demod_reg_address addr;
          const char *regName = "";

          switch (regId)
          {
          case '0':
            addr = demod_reg_address::ADDR_DCOI;
            regName = "DCOI";
            break;
          case '1':
            addr = demod_reg_address::ADDR_DCOQ;
            regName = "DCOQ";
            break;
          case '2':
            addr = demod_reg_address::ADDR_HD2IX;
            regName = "HD2IX";
            break;
          case '3':
            addr = demod_reg_address::ADDR_HD2IY;
            regName = "HD2IY";
            break;
          case '4':
            addr = demod_reg_address::ADDR_HD2QX;
            regName = "HD2QX";
            break;
          case '5':
            addr = demod_reg_address::ADDR_HD2QY;
            regName = "HD2QY";
            break;
          case '6':
            addr = demod_reg_address::ADDR_HD3IX;
            regName = "HD3IX";
            break;
          case '7':
            addr = demod_reg_address::ADDR_HD3IY;
            regName = "HD3IY";
            break;
          case '8':
            addr = demod_reg_address::ADDR_HD3QX;
            regName = "HD3QX";
            break;
          case '9':
            addr = demod_reg_address::ADDR_HD3QY;
            regName = "HD3QY";
            break;
          case 'A': // GERR - bits [5:0] of ADDR_GERR_IP3CC (0x11)
          {
            bool wasRunning = sampling_enabled;
            if (wasRunning)
              stop_sampling();
            reinit_spi_software_mode();
            uint8_t current = demod_read_reg(demod_reg_address::ADDR_GERR_IP3CC);
            current = (current & 0xC0) | (value & 0x3F); // Keep bits [7:6], set bits [5:0]
            demod_write_reg(demod_reg_address::ADDR_GERR_IP3CC, current);
            reinit_spi_hw_adc_mode();
            if (wasRunning)
              start_sampling();
            if (SerialUSB)
            {
              SerialUSB.print("GERR set to: ");
              SerialUSB.println(value);
            }
            return;
          }
          case 'B': // DEMOD_ATT - bits [4:0] of ADDR_ATT_IP3IC (0x10)
          {
            bool wasRunning = sampling_enabled;
            if (wasRunning)
              stop_sampling();
            reinit_spi_software_mode();
            uint8_t current = demod_read_reg(demod_reg_address::ADDR_ATT_IP3IC);
            current = (current & 0xE0) | (value & 0x1F); // Keep bits [7:5], set bits [4:0]
            demod_write_reg(demod_reg_address::ADDR_ATT_IP3IC, current);
            reinit_spi_hw_adc_mode();
            if (wasRunning)
              start_sampling();
            if (SerialUSB)
            {
              SerialUSB.print("DEMOD_ATT set to: ");
              SerialUSB.println(value);
            }
            return;
          }
          default:
            return; // Invalid register ID
          }

          // Temporarily stop sampling for SPI access
          bool wasRunning = sampling_enabled;
          if (wasRunning)
          {
            stop_sampling();
          }
          reinit_spi_software_mode();
          demod_write_reg(addr, value);
          reinit_spi_hw_adc_mode();
          if (wasRunning)
          {
            start_sampling();
          }
          if (SerialUSB)
          {
            SerialUSB.print(regName);
            SerialUSB.print(" set to: ");
            SerialUSB.println(value);
          }
        }
      }
    }
  }

  switch (currentState)
  {
  case STATE_ADC_SAMPLING:
    // Continue sending samples while in this state
    send_samples_usb();

    break;

  case STATE_WAIT_AFTER_ADC:
    // Wait for 1 second
    if (now - stateStartTime >= 100)
    {
      if (SerialUSB)
      {
        SerialUSB.println("--- Starting DVGA write phase (2s) ---");
      }

      // Reconfigure SPI for DVGA (software mode)
      reinit_spi_software_mode();
      if (gainReceived)
      {
        currentState = STATE_DIRECT_DVGA_WRITE;
        gainReceived = false; // reset flag
      }
      else
      {
        currentState = STATE_DVGA_WRITING;
      }
      stateStartTime = now;
    }
    break;

  case STATE_DVGA_WRITING:
    // Write to DVGA continuously - sweep through all gain values once
    {
      static uint32_t lastDvgaWrite = 0;
      static bool sweepComplete = false;

      if (now - lastDvgaWrite >= 100) // Write every 100ms
      {
        dvga_set((uint8_t)dvgaGainValue);

        if (SerialUSB)
        {
          SerialUSB.print("DVGA gain set to: ");
          SerialUSB.print(dvgaGainValue);
          SerialUSB.println(dvgaParallelMode ? " (parallel)" : " (serial)");
        }

        // Cycle through gain values 0-63 in steps of 4
        dvgaGainValue = (dvgaGainValue + 4) & 0x1C;
        lastDvgaWrite = now;

        // Check if we've completed one full sweep (wrapped back to 0)
        if (dvgaGainValue == 0)
        {
          sweepComplete = true;
        }
      }

      // Exit when sweep is complete
      if (sweepComplete)
      {
        sweepComplete = false; // Reset for next time
        dvgaGainValue = 0;     // Reset gain value

        if (SerialUSB)
        {
          SerialUSB.println("--- DVGA sweep complete, restarting ADC ---");
        }

        // Reconfigure SPI for ADC hardware mode
        reinit_spi_hw_adc_mode();

        // Start sampling again
        start_sampling();

        currentState = STATE_ADC_SAMPLING;
        stateStartTime = now;
      }
    }
    break;

  case STATE_DIRECT_DVGA_WRITE:
    // Direct DVGA write state - switch SPI, write, switch back
    {
      // Switch to software SPI mode for DVGA write (only needed for serial mode)
      if (!dvgaParallelMode)
      {
        reinit_spi_software_mode();
      }

      // Write the gain value (uses serial or parallel based on mode)
      dvga_set((uint8_t)receivedGainValue);
      delay(100);

      if (SerialUSB)
      {
        SerialUSB.print("DVGA set to: ");
        SerialUSB.print(receivedGainValue);
        SerialUSB.print(dvgaParallelMode ? " (parallel)" : " (serial)");
        SerialUSB.println();
      }

      // Switch back to hardware SPI mode for ADC
      reinit_spi_hw_adc_mode();

      // Resume sampling if it was running before
      if (samplingWasRunning)
      {
        start_sampling();
      }

      // Return to ADC sampling state
      currentState = STATE_ADC_SAMPLING;
      stateStartTime = now;
    }
    break;

  case STATE_DIRECT_ATT_WRITE:
    // Direct attenuator write state - switch SPI, write, switch back
    {
      // Switch to software SPI mode for attenuator write
      reinit_spi_software_mode();

      // Write the attenuator value (uses serial or parallel based on mode)
      att_set((uint8_t)receivedAttValue);
      delay(100);

      if (SerialUSB)
      {
        SerialUSB.print("Attenuator set to: ");
        SerialUSB.print(receivedAttValue);
        SerialUSB.print(attParallelMode ? " (parallel)" : " (serial)");
        SerialUSB.println();
      }

      attReceived = false; // Reset flag

      // Switch back to hardware SPI mode for ADC
      reinit_spi_hw_adc_mode();

      // Resume sampling if it was running before
      if (samplingWasRunning)
      {
        start_sampling();
      }

      // Return to ADC sampling state
      currentState = STATE_ADC_SAMPLING;
      stateStartTime = now;
    }
    break;
  }
}
