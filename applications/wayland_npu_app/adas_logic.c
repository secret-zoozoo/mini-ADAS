#include "adas_logic.h"
#include "nc_cnn_network_includes.h"
#include <math.h>
#include <stdio.h>

// Thresholds
#define FCW_WIDTH_THRESHOLD_PERCENT 0.30f // 30% of screen width
#define BSD_WIDTH_THRESHOLD_PERCENT 0.10f // 10% of screen width
#define BSD_PROB_THRESHOLD 0.50f
#define LDW_LANE_CENTER_THRESHOLD 50.0f // pixels from center

// Global Alert State (can be used for filtering/debounce)
#ifndef LANE_MAX_NUM
#define LANE_MAX_NUM 4
#endif

// Global Alert State (can be used for filtering/debounce)
// static int g_fcw_state = 0;
// static int g_ldw_state = 0;
// // static int g_bsd_state = 0;

void check_adas_rules(int cam_ch, stCnnPostprocessingResults *result,
                      AdaAlertState *alerts) {
  // Initialize alerts
  if (alerts) {
    alerts->fcw = 0;
    alerts->ldw = 0;
    alerts->bsd_left = 0;
    alerts->bsd_right = 0;
  }

  if (cam_ch == 0) // Front Camera
  {
    // 1. FCW Logic
    for (int i = 0; i < MAX_CNN_RESULT_CNT_OF_CLASS; i++) {
      // Assuming Class 2 is CAR, 5 is BUS, 7 is TRUCK (COCO dataset)
      // But we need to check the actual class ID map. YOLOv8 COCO: 2=car,
      // 5=bus, 7=truck. Let's iterate all classes or specific ones.
      // Implementation Plan says: "bbox.w > 30% of screen"

      // Check 'Car' class (id 2)
      if (result->class_objs[2].obj_cnt > 0) {
        stObjInfo *obj = &result->class_objs[2].objs[i];
        // bbox.w is normalized or pixel? YOLO postprocess usually gives pixel
        // relative to target_w. We should assume pixel values or check YOLO
        // postprocess. If pixel, we need to know screen width. Let's assume
        // normalized input (0.0-1.0) was converted to pixel coordinates in
        // postprocess. NPU_INPUT_WIDTH is 640. 30% is ~192 pixels.

        if (obj->prob > 0.5f &&
            obj->bbox.w > (NPU_INPUT_WIDTH * FCW_WIDTH_THRESHOLD_PERCENT)) {
          alerts->fcw = 1;
          printf("[ADAS] FCW Alert! Car Width: %.1f\n", obj->bbox.w);
        }
      }
    }

    // 2. LDW Logic
    // Iterate lanes. If any point is close to center X (320).
    for (int i = 0; i < LANE_MAX_NUM; i++) {
      if (result->lane_det[i].point_cnt > 0) {
        // Check bottom-most points (closer to car)
        int check_idx = result->lane_det[i].point_cnt - 1; // Last point
        if (check_idx >= 0) {
          float lx = result->lane_det[i].point[check_idx].x;
          float ly = result->lane_det[i].point[check_idx].y;

          // If y is close to bottom and x is close to center
          if (ly > NPU_INPUT_HEIGHT * 0.8f &&
              fabs(lx - (NPU_INPUT_WIDTH / 2.0f)) < LDW_LANE_CENTER_THRESHOLD) {
            alerts->ldw = 1;
            // printf("[ADAS] LDW Alert! Lane X: %.1f\n", lx);
          }
        }
      }
    }
  } else if (cam_ch == 1) // Rear Camera
  {
    // 3. BSD / LCA Logic (Left/Right Separation)
    // Define "Ego Lane" area in the center. Vehicles outside this are
    // candidates. NPU_INPUT_WIDTH is 640. Center is 320. Assume Ego Lane Width
    // is roughly 200px (depends on camera). Left Zone: x < 220, Right Zone: x >
    // 420

    float left_boundary = (NPU_INPUT_WIDTH / 2.0f) - 100.0f;
    float right_boundary = (NPU_INPUT_WIDTH / 2.0f) + 100.0f;

    for (int c = 0; c < 80; c++) { // Iterate all classes
      if (c != 2 && c != 5 && c != 7)
        continue; // Only vehicles (Car, Bus, Truck)

      for (int i = 0; i < result->class_objs[c].obj_cnt; i++) {
        stObjInfo *obj = &result->class_objs[c].objs[i];

        // Skip low probability
        if (obj->prob < BSD_PROB_THRESHOLD)
          continue;

        // Calculate Center X of the vehicle
        float obj_cx = obj->bbox.x; // bbox.x is usually center x

        // Check Vehicle Size (Too Close?)
        if (obj->bbox.w > (NPU_INPUT_WIDTH * BSD_WIDTH_THRESHOLD_PERCENT)) {

          // Check Left Side
          if (obj_cx < left_boundary) {
            alerts->bsd_left = 1;
            // printf("[ADAS] Left Blind Spot Alert! Width: %.1f\n",
            // obj->bbox.w);
          }
          // Check Right Side
          else if (obj_cx > right_boundary) {
            alerts->bsd_right = 1;
            // printf("[ADAS] Right Blind Spot Alert! Width: %.1f\n",
            // obj->bbox.w);
          }
        }
      }
    }
  }
}
