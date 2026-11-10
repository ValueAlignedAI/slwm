#include <stdint.h>
#include <stddef.h>
/* Castagnoli CRC; one table is built per call to keep the library reentrant. */
uint32_t slwm_crc32c(const unsigned char *data, size_t length) {
    uint32_t table[256];
    for (unsigned i=0; i<256; ++i) {
        uint32_t value=i;
        for (unsigned j=0; j<8; ++j) value=(value>>1)^((value&1)?0x82f63b78U:0);
        table[i]=value;
    }
    uint32_t crc=0xffffffffU;
    for (size_t i=0; i<length; ++i) crc=(crc>>8)^table[(crc^data[i])&255];
    return crc^0xffffffffU;
}
