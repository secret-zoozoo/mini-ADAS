#ifndef __ADAS_LOGIC_H__
#define __ADAS_LOGIC_H__

// Include the header that defines stCnnPostprocessingResults
#include "nc_cnn_aiware_runtime.h"
// Or check where stCnnPostprocessingResults is defined.
// It is likely in nc_cnn_structs.h or similar if separated.
// But wayland_npu_app.c uses nc_cnn_aiware_runtime.h

typedef struct {
  int fcw;
  int ldw;
  int bsd_left;
  int bsd_right;
} AdaAlertState;

// NPU Input Size (Assuming 640x384 standard for these models)
#ifndef NPU_INPUT_WIDTH
#define NPU_INPUT_WIDTH 640
#endif
#ifndef NPU_INPUT_HEIGHT
#define NPU_INPUT_HEIGHT 384
#endif

void check_adas_rules(int cam_ch, stCnnPostprocessingResults *result,
                      AdaAlertState *alerts);

#endif
