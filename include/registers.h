#ifndef REGISTERS_H
#define REGISTERS_H

#include <Arduino.h>
#include <stdint.h>
#include "pins.h"
typedef enum : uint8_t
{
  ADDR_IM3QY = 0x00,
  ADDR_IM3QX = 0x01,
  ADDR_IM3IY = 0x02,
  ADDR_IM3IX = 0x03,
  ADDR_IM2QX = 0x04,
  ADDR_IM2IX = 0x05,
  ADDR_HD3QY = 0x06,
  ADDR_HD3QX = 0x07,
  ADDR_HD3IY = 0x08,
  ADDR_HD3IX = 0x09,
  ADDR_HD2QY = 0x0A,
  ADDR_HD2QX = 0x0B,
  ADDR_HD2IY = 0x0C,
  ADDR_HD2IX = 0x0D,
  ADDR_DCOI = 0x0E,
  ADDR_DCOQ = 0x0F,
  ADDR_ATT_IP3IC = 0x10,
  ADDR_GERR_IP3CC = 0x11,
  ADDR_LVCM_CF1 = 0x12,
  ADDR_BAND_LF1_CF2 = 0x13,
  ADDR_PHA_8 = 0x14,
  ADDR_PHA_0_AMPG_AMPCC_AMPIC = 0x15,
  ADDR_SRST_SDO_MODE = 0x16,
  ADDR_CHIPID_RFSW = 0x17
} demod_reg_address;


typedef enum {
  REG_ATTN = 0x00,
  REG_CONFIG = 0x01,
  REG_RFAREG = 0x02,
} att_reg_address;


extern const uint8_t REG_DEFAULT16[0x18];

uint8_t demod_read_reg(demod_reg_address addr);
void demod_write_reg(demod_reg_address addr, uint8_t data);
void demod_set_default();
void demod_verify_all();
#endif
