#include <Arduino.h>
#include <SPI.h>

// --- Pins ---
constexpr uint8_t CONVST_PIN = 2;   // D2 = PB25 = TIOA0
constexpr uint8_t CS1_PIN    = 10;  // SPI CS0 (hardware)
constexpr uint8_t CS2_PIN    = 4;   // SPI CS1 (hardware)

// --- Timing targets ---
constexpr uint32_t CONVST_LOW_NS  = 100;
constexpr uint32_t CONVST_HIGH_NS = 12400;   // 12.5us period => 80kSPS
constexpr uint32_t TCONV_MAX_NS   = 8800;    // ADS8867 tconv max

// SPI clock (keep <= device limit; you said this works)
constexpr uint32_t SPI_SCK_HZ     = 14000000;

// --------- Ring buffer for samples ---------
// Each sample = 2x uint16 => 4 bytes
struct SamplePair {
  uint16_t a1;
  uint16_t a2;
};

// Power-of-two size for fast wrap
constexpr uint32_t RB_BITS = 13;                 // 2^13 = 8192 samples
constexpr uint32_t RB_SIZE = (1u << RB_BITS);
constexpr uint32_t RB_MASK = RB_SIZE - 1;

SamplePair rb[RB_SIZE];
volatile uint32_t rb_head = 0;
volatile uint32_t rb_tail = 0;
volatile uint32_t rb_overflow = 0;
volatile uint32_t seq = 0;

// ---------- Helpers ----------
static inline uint32_t ns_to_ticks(uint32_t ns, uint32_t clk_hz) {
  return (uint32_t)(((uint64_t)ns * clk_hz + 500000000ULL) / 1000000000ULL);
}

// PCS codes for variable select mode
static constexpr uint8_t PCS_NPCS0 = 0; // D10
static constexpr uint8_t PCS_NPCS1 = 1; // D4

static inline uint16_t spi0_read16(uint8_t pcs_code) {
  while (!(SPI0->SPI_SR & SPI_SR_TDRE)) {}
  SPI0->SPI_TDR = (uint32_t)(0x0000u) | ((uint32_t)(pcs_code & 0xF) << 16);
  while (!(SPI0->SPI_SR & SPI_SR_RDRF)) {}
  return (uint16_t)(SPI0->SPI_RDR & 0xFFFFu);
}

static void setup_spi_hw_cs() {
  SPI.begin();

  pmc_enable_periph_clk(ID_SPI0);
  pmc_enable_periph_clk(ID_PIOA);

  // Hand PA28/PA29 (NPCS0/NPCS1) to peripheral A
  const uint32_t PA28 = (1u << 28);
  const uint32_t PA29 = (1u << 29);
  PIOA->PIO_PDR  = PA28 | PA29;
  PIOA->PIO_ABSR &= ~(PA28 | PA29);
  PIOA->PIO_PUDR = PA28 | PA29;

  SPI0->SPI_CR = SPI_CR_SWRST;
  SPI0->SPI_CR = SPI_CR_SWRST;

  SPI0->SPI_MR = SPI_MR_MSTR | SPI_MR_MODFDIS | SPI_MR_PS | SPI_MR_DLYBCS(6);

  uint8_t scbr = (uint8_t)(VARIANT_MCK / SPI_SCK_HZ);
  if (scbr < 1) scbr = 1;
  if (scbr > 255) scbr = 255;

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

static void setup_convst_and_timer_trigger() {
  const uint32_t tc_clk_hz = VARIANT_MCK / 2; // 42 MHz

  uint32_t low_ticks  = ns_to_ticks(CONVST_LOW_NS,  tc_clk_hz);
  uint32_t high_ticks = ns_to_ticks(CONVST_HIGH_NS, tc_clk_hz);
  if (low_ticks < 2) low_ticks = 2;
  if (high_ticks < 2) high_ticks = 2;
  const uint32_t period_ticks = low_ticks + high_ticks;

  uint32_t tconv_ticks = ns_to_ticks(TCONV_MAX_NS, tc_clk_hz);
  if (tconv_ticks < 1) tconv_ticks = 1;

  // CONVST rising at RA0=low_ticks, trigger read at low_ticks + tconv_ticks
  const uint32_t read_tick = low_ticks + tconv_ticks;

  pmc_enable_periph_clk(ID_TC0);
  pmc_enable_periph_clk(ID_TC1);

  // Route D2 (PB25) to TIOA0 (Peripheral B)
  pmc_enable_periph_clk(ID_PIOB);
  PIOB->PIO_PDR  |= PIO_PDR_P25;
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
  TC0->TC_CHANNEL[1].TC_RA  = read_tick;
  TC0->TC_CHANNEL[1].TC_RC  = period_ticks;
  TC0->TC_CHANNEL[1].TC_IER = TC_IER_CPAS;

  NVIC_SetPriority(TC1_IRQn, 0);
  NVIC_EnableIRQ(TC1_IRQn);

  TC0->TC_CHANNEL[0].TC_CCR = TC_CCR_CLKEN;
  TC0->TC_CHANNEL[1].TC_CCR = TC_CCR_CLKEN;
  TC0->TC_BCR = TC_BCR_SYNC;
}

// ISR: acquire and push into ring buffer (NO serial here!)
void TC1_Handler() {
  uint32_t sr = TC0->TC_CHANNEL[1].TC_SR;
  if (sr & TC_SR_CPAS) {
    SamplePair s;
    s.a1 = spi0_read16(PCS_NPCS0);
    s.a2 = spi0_read16(PCS_NPCS1);

    uint32_t head = rb_head;
    uint32_t next = (head + 1) & RB_MASK;

    if (next == rb_tail) {
      rb_overflow++;
    } else {
      rb[head] = s;
       __DMB();
      rb_head = next;
    }
    seq++;
  }
}

// ---------- USB packet format ----------
// Header: 'A''D''C''\n' (4 bytes)
// uint32_t seq_end   (4 bytes) : sequence number at end of this packet
// uint16_t count     (2 bytes) : number of samples in this packet
// Payload: count * SamplePair (count*4 bytes)
// Total header = 10 bytes
static void send_samples_usb() {
  if (!SerialUSB) return;

  constexpr uint16_t MAX_SAMPLES_PER_PACKET = 12; // 58 bytes total per packet

  // Snapshot head/tail
  uint32_t tail = rb_tail;
  uint32_t head = rb_head;
  if (tail == head) return;

  uint32_t available = (head - tail) & RB_MASK;
  uint16_t n = (available > MAX_SAMPLES_PER_PACKET) ? MAX_SAMPLES_PER_PACKET : (uint16_t)available;

  uint32_t to_end = RB_SIZE - tail;
  if (n > to_end) n = (uint16_t)to_end;

  // Header
  uint8_t hdr[10];
  hdr[0] = 'A'; hdr[1] = 'D'; hdr[2] = 'C'; hdr[3] = '\n';
  uint32_t seq_end = seq;
  memcpy(&hdr[4], &seq_end, 4);
  memcpy(&hdr[8], &n, 2);

  // Write header + payload
  SerialUSB.write(hdr, 10);
  SerialUSB.write((const uint8_t*)&rb[tail], n * sizeof(SamplePair));

  // Publish tail
  rb_tail = (tail + n) & RB_MASK;
}

void setup() {
  // Use the Native USB port (SerialUSB). Baud ignored.
  SerialUSB.begin(115200);

  setup_spi_hw_cs();
  setup_convst_and_timer_trigger();
}

void loop() {
  send_samples_usb();

  // Optional: if you want to monitor overflow occasionally (don’t spam)
  // static uint32_t last_ms = 0;
  // if (millis() - last_ms > 1000) {
  //   last_ms = millis();
  //   uint32_t ov = rb_overflow;
  //   // Send a tiny ASCII status line once per second (safe)
  //   SerialUSB.print("ovf=");
  //   SerialUSB.println(ov);
  //}
}
