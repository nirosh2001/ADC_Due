#include <Arduino.h>
#include <SPI.h>
#include "registers.h"

const uint8_t REG_DEFAULT16[0x18] = {
    /* 0x00 */ 0x80,
    /* 0x01 */ 0x80,
    /* 0x02 */ 0x80,
    /* 0x03 */ 0x80,
    /* 0x04 */ 0x80,
    /* 0x05 */ 0x80,
    /* 0x06 */ 0x80,
    /* 0x07 */ 0x80,
    /* 0x08 */ 0x80,
    /* 0x09 */ 0x80,
    /* 0x0A */ 0x80,
    /* 0x0B */ 0x80,
    /* 0x0C */ 0x80,
    /* 0x0D */ 0x80,
    /* 0x0E */ 0x80,
    /* 0x0F */ 0x80,
    /* 0x10 */ 0x4,
    /* 0x11 */ 0x82,
    /* 0x12 */ 0x48,
    /* 0x13 */ 0xE3,
    /* 0x14 */ 0x80,
    /* 0x15 */ 0x6A,
    /* 0x16 */ 0xF0,
    /* 0x17 */ 0x01};


static inline void demod_cs_low() { digitalWrite(DEMOD_CS_PIN, LOW); }
static inline void demod_cs_high() { digitalWrite(DEMOD_CS_PIN, HIGH); }

uint8_t demod_read_reg(demod_reg_address addr)
{
  demod_cs_low();
  SPI.beginTransaction(SPISettings(100000, MSBFIRST, SPI_MODE0));
  SPI.transfer(0x80 | (uint8_t)addr);
  delay(10);
  uint8_t msb = SPI.transfer(0x00);
  SPI.endTransaction();
  demod_cs_high();

  return msb;
}

void demod_write_reg(demod_reg_address addr, uint8_t data)
{
  SPI.beginTransaction(SPISettings(100000, MSBFIRST, SPI_MODE0));
  demod_cs_low();
  SPI.transfer((uint8_t)addr); // write (bit7 = 0)  // MSB
  SPI.transfer(data); // LSB
  demod_cs_high();
  SPI.endTransaction();
}

void demod_set_default()
{

  for (uint8_t a = ADDR_IM3QY; a <= ADDR_CHIPID_RFSW; a++)
  {
    demod_write_reg((demod_reg_address)a, REG_DEFAULT16[a]);
    delay(10);
  }
  return;
}

void demod_verify_all()
{
  for (uint8_t a = ADDR_IM3QY; a <= ADDR_CHIPID_RFSW; a++)
  {
    uint8_t val = demod_read_reg((demod_reg_address)a);
    Serial.print("address:");
    Serial.print((uint8_t)a, HEX);
    Serial.print(":=");
    Serial.println(val, HEX);
    delay(10);
  }
  return;
}