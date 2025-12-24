#!/bin/sh

if [ ! $APP_PATH ]; then
APP_PATH=.
fi

# vcodec clk divide
# devmem 0x6004600 32 0x06      # CODEC_BPU_CLKDIV    (PLL : 2400 Mhz, MAX : 400 Mhz)
# devmem 0x6004604 32 0x04      # CODEC_CORE_CLKDIV   (PLL : 2000 Mhz, MAX : 500 Mhz)
# devmem 0x6004608 32 0x06      # HWA_AXI_CLKDIV      (PLL : 2000 Mhz, MAX : 500 Mhz)
# devmem 0x600460c 32 0x10      # HWA_APB_CLKDIV      (PLL : 2000 Mhz, MAX : 200 Mhz)

export LD_LIBRARY_PATH=${APP_PATH}/misc/lib
# export VIDEO_CODEC_FIRMWARE_FILE=/lib/wave_fw/chagall.bin

insmod ${APP_PATH}/../modules/vpu.ko

${APP_PATH}/app_vcodec