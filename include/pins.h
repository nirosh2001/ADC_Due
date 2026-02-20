#ifndef PINS_H
#define PINS_H

// attenuator
#define LE 3
#define PBAR 9

// Attenuator parallel mode pins (D0-D6)
#define ATT_D0 53
#define ATT_D1 52
#define ATT_D2 51
#define ATT_D3 50
#define ATT_D4 49
#define ATT_D5 48
#define ATT_D6 47

// demodulator
#define DEMOD_CS_PIN 8
#define RFSW 26 // 3.3

// DVGA IC 1
#define OE 7
#define SENB 6
#define DVGA_CS_PIN 5
#define DVGA_DENA 32
#define DVGA_DENB 31

// DVGA IC 1 parallel mode pins (D0-D5)
#define DVGA_D0 5
#define DVGA_D1 37
#define DVGA_D2 36
#define DVGA_D3 35
#define DVGA_D4 34
#define DVGA_D5 33

// DVGA IC 2
#define DVGAI_OE 11
#define DVGAI_SENB 13
#define DVGAI_CS_PIN 12
#define DVGAI_DENA 40
#define DVGAI_DENB 39

// DVGA IC 2 parallel mode pins (D0-D5)
#define DVGAI_D0 12
#define DVGAI_D1 45
#define DVGAI_D2 44
#define DVGAI_D3 43
#define DVGAI_D4 42
#define DVGAI_D5 41

#define LNA_1 28
#define LNA_2 29

#endif