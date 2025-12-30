/*
********************************************************************************
* Copyright (C) 2021 NEXTCHIP Inc. All rights reserved.
* This software is the confidential and proprietary information of
* NEXTCHIP, Inc. ("Confidential Information"). You shall not disclose such
* Confidential Information and shall use it only in accordance with
* the terms of the license agreement you entered into with NEXTCHIP.
********************************************************************************

********************************************************************************
* @file    : wayland_npu_app.c
*
* @brief   : wayland_npu application
*
* @author  : Software Development Team.  NextChip Inc.
*
* @date    : 2024.04.26.
*
* @version : 1.0.0
********************************************************************************
* @note
*
********************************************************************************
*/

/*
********************************************************************************
*               INCLUDES
********************************************************************************
*/

#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <SOIL.h>
#include <arm_neon.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/input.h>
#include <linux/videodev2.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "nc_app_config_parser.h"
#include "nc_cnn_aiware_runtime.h"
#include "nc_cnn_communicator.h"
#include "nc_cnn_worker_for_postprocess.h"
#include "nc_neon.h"
#include "nc_opengl_init.h"
#include "nc_opengl_interface.h"
#include "nc_opengl_shader.h"
#include "nc_ts_fsync_flipflop_buffers.h"
#include "v4l2_interface.h"
#include "wayland_egl.h"
#ifdef AIWARE_DEVICE_SUPPORTED
#include "aiware/runtime/c/aiwaredevice.h"
#endif
#include "nc_opengl_ttf_font.h"
#ifdef USE_BYTETRACK
#include "nc_cnn_tracker.h"
#endif

#ifdef USE_8MP_VI
#include "nc_dmabuf_ctrl_helper.h"
#include "nc_dsr_helper.h"
#include "nc_dsr_set.h"
#endif

#ifdef USE_ADAS_LD
#include "ADAS_LD_Lib.h"
#endif

/*
********************************************************************************
*               DEFINES
********************************************************************************
*/

#define VIS0_MAX_CH (1)
#define VIS1_MAX_CH (1)
#define VIDEO_MAX_CH (VIS0_MAX_CH + VIS1_MAX_CH)

#define VIDEO_BUFFER_NUM (3)

#ifdef USE_8MP_VI
#define VIDEO_WIDTH MAX_WIDTH_FOR_VDMA_CNN_DS
#define VIDEO_HEIGHT MAX_HEIGHT_FOR_VDMA_CNN_DS
#else
#define VIDEO_WIDTH NPU_INPUT_WIDTH
#define VIDEO_HEIGHT NPU_INPUT_HEIGHT
#endif

#define CAP_FPS (30)

#define MQ_NAME_CNN_BUF "/cnn_data"
#define DEV_FILE_DSR "/dev/dsr"

#ifdef SHOW_PELEE_SEG
#define NETWORK_FILE_PELEE_SEG                                                 \
  "misc/networks/peleeseg/peleeseg_640x384_apache6sr250_aiw4939.aiwbin"
#endif
#ifdef SHOW_PELEE_DETECT
#define NETWORK_FILE_PELEE_DET                                                 \
  "misc/networks/peleeDet/"                                                    \
  "peleedet_10class_640x384_apache6sr250_aiw4939.aiwbin"
#endif
#ifdef SHOW_YOLOV5_DETECT
#define NETWORK_FILE_YOLOV5_DET                                                \
  "misc/networks/Yolov5s/yolov5s_coco_640x384_apache6sr250_aiw4939.aiwbin"
#endif
#ifdef SHOW_YOLOV8_DETECT
#define NETWORK_FILE_YOLOV8_DET                                                \
  "misc/networks/Yolov8/yolov8s_coco_640x384_apache6sr250_aiw4939.aiwbin"
#endif
#ifdef SHOW_UFLD_LANE
#define NETWORK_FILE_UFLD_LANE "misc/networks/ufld/ufld_a6sr250_aiw4939.aiwbin"
#endif
#ifdef SHOW_TRI_CHIMERA
#define NETWORK_FILE_TRI_CHIMERA                                               \
  "misc/networks/trichimera/trichimera_640x384_a6sr_aiw4939.aiwbin"
#endif
#define COLOR_PALETTE_CNT (10)
typedef enum {
  RGBA_R = 0,
  RGBA_G,
  RGBA_B,
  RGBA_A,
  RGBA_CNT,
} E_RGBA_IDX;

typedef enum {
  DRIVING_MODE = 0,
  PARKING_MODE = 1,
} E_MODE;

/* ------------------------- Collision Detection Configuration
 * ------------------------- */
#define FRONT_CH (0)
#define REAR_CH (1)

/* Fast approach detection thresholds (tunable) */
#define WARN_MIN_PREV_AREA (2000.0f) // Ignore very small boxes (noise)
#define WARN_GROWTH_FACTOR (1.35f)   // Alert if area grows 1.35x or more
#define WARN_MAX_DT_SEC (0.25f)      // Within 0.25 seconds = fast approach
#define WARN_MIN_AREA (8000.0f)      // Don't warn if current area too small

/* Segmentation class IDs (from TRI_CHIMERA model) */
#define SEG_CLASS_ROAD 0         // Drivable area (safe)
#define SEG_CLASS_SIDEWALK 1     // Sidewalk
#define SEG_CLASS_BUILDING 2     // Building/Wall (danger)
#define SEG_CLASS_WALL 3         // Wall (danger)
#define SEG_CLASS_VEHICLE 4      // Vehicle
#define SEG_WALL_THRESHOLD 0.30f // 30% wall pixels = warning

/*    don't modify  */
#define MAX_WIDTH_FOR_VDMA_CNN_DS (1920)
#define MAX_HEIGHT_FOR_VDMA_CNN_DS (1080)
/********************/

/*
********************************************************************************
*               VARIABLE DECLARATIONS
********************************************************************************
*/

st_npu_input_info npu_input_info;
struct gl_npu_program g_npu_prog;
GLuint g_seg_texture;
struct gl_font_program g_font_prog;
Font font_38;
Font font_24;

unsigned char *image_data;
static int running = 1;

// Mode Switching
volatile sig_atomic_t g_current_mode = DRIVING_MODE;

// Collision Detection State (per channel)
static float g_prev_max_area[VIDEO_MAX_CH] = {0};
static uint64_t g_prev_ts_us[VIDEO_MAX_CH] = {0};

st_nc_v4l2_config v4l2_config[VIDEO_MAX_CH];

struct viewport g_viewport[VIDEO_MAX_CH];

#ifdef USE_8MP_VI
size_t BUFF_SIZE;
int dsr_fd;
DSR_Data_t dsr_info;
dma_alloc_info dma_info;
st_nc_dsr_config dsr_config;
img_input_config dsr_input_config;
img_output_config dsr_output_config;
#endif
/*
********************************************************************************
*               FUNCTION DECLARATIONS
********************************************************************************
*/
bool is_point_in_trapezoid(float px, float py, float x1, float y1, float x2,
                           float y2, float x3, float y3, float x4, float y4);
bool get_largest_bbox(const stCnnPostprocessingResults *det_result,
                      const stObjDrawInfo *draw_info, stObjInfo *out_obj,
                      float *out_area, int *out_class_id);
void draw_thick_red_bbox(const stObjInfo *obj, struct gl_npu_program prog);
bool is_fast_approach(uint32_t ch, float cur_area, uint64_t now_us);
bool check_wall_in_guideline(const uint8_t *seg_data, int seg_width,
                             int seg_height, float x1, float y1, float x2,
                             float y2, float x3, float y3, float x4, float y4);

/* Render sub-functions */
void render_camera_frames(struct window *window, int *networkOrder);
void render_detection_results(struct window *window);
void render_parking_guidelines(void);
void render_ui_overlays(const char *buftext);

/*
********************************************************************************
*               FUNCTION DEFINITIONS
********************************************************************************
*/
#ifdef USE_8MP_VI
static int dsr_init(void) {
  int input_fd = 0, output_fd = 0;
  long page_size = sysconf(_SC_PAGESIZE);

  dsr_input_config.format = IMG_FORMAT_RGB888;
  dsr_input_config.width = MAX_WIDTH_FOR_VDMA_CNN_DS;
  dsr_input_config.height = MAX_HEIGHT_FOR_VDMA_CNN_DS;
  dsr_output_config.format = IMG_FORMAT_RGB888;

  BUFF_SIZE = dsr_input_config.width * dsr_input_config.height * 3;
  BUFF_SIZE = ((BUFF_SIZE + page_size - 1) / page_size) * page_size;

  if (dsr_config_downscale(&dsr_config, 1, NPU_INPUT_WIDTH, NPU_INPUT_HEIGHT) <
      0) {
    perror("Error: DSR config fail");
    return -1;
  }

  if (open_device_and_dma_buffers(DEV_FILE_DSR, &dsr_fd, &input_fd, &output_fd,
                                  BUFF_SIZE) < 0) {
    perror("Error: DSR open fail");
    return -1;
  }

  if (dsr_setup_buffer(&dsr_info, dsr_fd, &dma_info, input_fd, output_fd,
                       BUFF_SIZE) < 0) {
    perror("Error: DSR setup fail");
    return -1;
  }

  return 0;
}

static int dsr_deinit(void) {
  nc_dmabuf_ctrl_end_cpu_access(dma_info.dmabuf_fd_in);
  nc_dmabuf_ctrl_end_cpu_access(dma_info.dmabuf_fd_out);

  nc_dmabuf_ctrl_free_dma_fd(dma_info.dmabuf_fd_in);
  nc_dmabuf_ctrl_free_dma_fd(dma_info.dmabuf_fd_out);

  nc_dmabuf_ctrl_close();

  for (int i = 0; i < BUFF_NUM; i++) {
    if (dsr_info.dsr_in_buf[i] != MAP_FAILED) {
      munmap(dsr_info.dsr_in_buf[i], BUFF_SIZE);
    }
    if (dsr_info.dsr_out_buf[i] != MAP_FAILED) {
      munmap(dsr_info.dsr_out_buf[i], BUFF_SIZE);
    }
  }

  dsr_device_deinit(&dsr_fd);

  return 0;
}
#endif

void set_viewport_config(void) {
#if (VIDEO_MAX_CH == 1)
  // full screen view
  g_viewport[0].x = 0;
  g_viewport[0].y = 0;
  g_viewport[0].width = WINDOW_WIDTH;
  g_viewport[0].height = WINDOW_HEIGHT;
#elif (VIDEO_MAX_CH == 2)
  // Split Screen (Top/Bottom) - Centered with side margins
  // Camera aspect ratio: 16:9
  // Display: 1920x1080, use center 1440 width (720 per camera)
  int camera_width = 1440; // Centered width for both cameras
  int margin_x = (WINDOW_WIDTH - camera_width) / 2; // Side margins

  // Top Viewport (Front Camera - CH0)
  g_viewport[0].x = margin_x;
  g_viewport[0].y = WINDOW_HEIGHT / 2;
  g_viewport[0].width = camera_width;
  g_viewport[0].height = WINDOW_HEIGHT / 2;

  // Bottom Viewport (Rear Camera - CH1)
  g_viewport[1].x = margin_x;
  g_viewport[1].y = 0;
  g_viewport[1].width = camera_width;
  g_viewport[1].height = WINDOW_HEIGHT / 2;
#elif (VIDEO_MAX_CH > 1)
  // quad view
  for (int i = 0; i < VIDEO_MAX_CH; i++) {
    g_viewport[i].width = WINDOW_WIDTH / 2;
    g_viewport[i].height = WINDOW_HEIGHT / 2;

    // Set the view position for each channel
    switch (i) {
    case 0:
      g_viewport[i].x = 0;
      g_viewport[i].y = WINDOW_HEIGHT / 2;
      break;
    case 1:
      g_viewport[i].x = WINDOW_WIDTH / 2;
      g_viewport[i].y = WINDOW_HEIGHT / 2;
      break;
    case 2:
      g_viewport[i].x = 0;
      g_viewport[i].y = 0;
      break;
    case 3:
      g_viewport[i].x = WINDOW_WIDTH / 2;
      g_viewport[i].y = 0;
      break;
    default:
      break;
    }
  }
#endif
}

void set_v4l2_config(void) {
  int i = 0, j = 0;

  for (i = 0; i < VIS0_MAX_CH; i++) {
    v4l2_config[i].video_buf.video_device_num = CNN_DEVICE_NUM(VISION0) + i;
    v4l2_config[i].video_buf.video_fd = -1;
#ifdef USE_8MP_VI
    v4l2_config[i].dma_mode = INTERLEAVE;
#else
    v4l2_config[i].dma_mode = PLANAR;
#endif
    v4l2_config[i].img_process = MODE_DS;
    v4l2_config[i].pixformat = V4L2_PIX_FMT_RGB24;
    v4l2_config[i].crop_x_start = 0;
    v4l2_config[i].crop_y_start = 0;
    v4l2_config[i].crop_width = 0;
    v4l2_config[i].crop_height = 0;
#ifdef USE_8MP_VI
    v4l2_config[i].ds_width = MAX_WIDTH_FOR_VDMA_CNN_DS;
    v4l2_config[i].ds_height = MAX_HEIGHT_FOR_VDMA_CNN_DS;
#else
    v4l2_config[i].ds_width = VIDEO_WIDTH;
    v4l2_config[i].ds_height = VIDEO_HEIGHT;
#endif
  }

  // Rear Camera Loop (VIS1)
  for (j = 0; j < VIS1_MAX_CH; j++) {
    // config_idx ensures we append after front cameras (if any)
    // If VIS0_MAX_CH is 0, this starts at 0.
    int config_idx = VIS0_MAX_CH + j;

    // Force Vision 1 (Rear Camera) Device Number
    // We use 'j' as offset: /dev/video60, /dev/video61...
    v4l2_config[config_idx].video_buf.video_device_num =
        CNN_DEVICE_NUM(VISION1) + j;

    printf("\n[DEBUG] =======================================\n");
    printf("[DEBUG] Configuring Camera Index: %d\n", config_idx);
    printf("[DEBUG] Target Device: /dev/video%d (VISION1)\n",
           v4l2_config[config_idx].video_buf.video_device_num);
    printf("[DEBUG] =======================================\n\n");

    v4l2_config[config_idx].video_buf.video_fd = -1;
#ifdef USE_8MP_VI
    v4l2_config[config_idx].dma_mode = INTERLEAVE;
#else
    v4l2_config[config_idx].dma_mode = PLANAR;
#endif
    // Revert: Use MODE_DS (Downscale) instead of CROP to avoid S_EXT_CTRLS
    // error
    v4l2_config[config_idx].img_process = MODE_DS;
    // [Fix] Hardware reports RGB3 (RGB24) in logs.
    // USE_8MP_VI default is YUYV, which causes S_EXT_CTRLS error.
    // We must force RGB24 to match hardware.
    v4l2_config[config_idx].pixformat = V4L2_PIX_FMT_RGB24;
    v4l2_config[config_idx].crop_x_start = 0;
    v4l2_config[config_idx].crop_y_start = 0;
    v4l2_config[config_idx].crop_width = 0;
    v4l2_config[config_idx].crop_height = 0;

    // Use default FHD resolution (1920x1080) for 2:1 downscaling from 4K
#ifdef USE_8MP_VI
    v4l2_config[config_idx].ds_width = MAX_WIDTH_FOR_VDMA_CNN_DS;
    v4l2_config[config_idx].ds_height = MAX_HEIGHT_FOR_VDMA_CNN_DS;
#else
    v4l2_config[config_idx].ds_width = VIDEO_WIDTH;
    v4l2_config[config_idx].ds_height = VIDEO_HEIGHT;
#endif
  }
}

int send_cnn_buf(uint8_t *ptr_cnn_buf, uint64_t time_stamp_us, uint32_t cam_ch,
                 E_NETWORK_UID net_id) {
  int ret = 0;
  stCnnData *cnn_data;
  struct mq_attr attr;
  attr.mq_maxmsg = MAX_MQ_MSG_CNT;
  attr.mq_msgsize = sizeof(stCnnData *);
  int oflag = O_WRONLY | O_CREAT;
  mqd_t mfd = mq_open(MQ_NAME_CNN_BUF, oflag, 0666, &attr);
  if (mfd == -1) {
    perror("mq open error");
    return -1;
  }

  cnn_data = (stCnnData *)malloc(sizeof(stCnnData));
  cnn_data->cam_ch = cam_ch;
  cnn_data->ptr_cnn_buf = ptr_cnn_buf;
  cnn_data->time_stamp_us = time_stamp_us;
  cnn_data->net_id = net_id;

  if ((ret = mq_send(mfd, (const char *)&cnn_data, attr.mq_msgsize, 1)) == -1) {
    printf("errno of mq_send = %d\n", errno);
  }
  mq_close(mfd);

  return ret;
}

int receive_cnn_buf(stCnnData **out_cnn_buf) {
  int ret = 0;
  struct mq_attr attr;
  attr.mq_maxmsg = MAX_MQ_MSG_CNT;
  attr.mq_msgsize = sizeof(stCnnData *);
  int oflag = O_RDONLY | O_CREAT;
  mqd_t mfd = mq_open(MQ_NAME_CNN_BUF, oflag, 0666, &attr);
  if (mfd == -1) {
    perror("mq open error");
    return -1;
  }

  if ((ret = (int32_t)mq_receive(mfd, (char *)out_cnn_buf, attr.mq_msgsize,
                                 NULL)) == -1) {
    printf("errno of mq_receive = %d\n", errno);
  }
  mq_close(mfd);

  return ret;
}

int v4l2_initialize(void) {
  for (int i = 0; i < VIDEO_MAX_CH; i++) {
    v4l2_config[i].video_buf.video_fd =
        nc_v4l2_open(v4l2_config[i].video_buf.video_device_num, true);
    if (v4l2_config[i].video_buf.video_fd == errno) {
      printf("[error] nc_v4l2_open() failure!\n");
    } else {
      if (nc_v4l2_init_device_and_stream_on(&v4l2_config[i], VIDEO_BUFFER_NUM) <
          0) {
        printf("[error] nc_v4l2_init_device_and_stream_on() failure!\n");
        return -1;
      }
    }
  }

  nc_v4l2_show_user_config(&v4l2_config[0], VIDEO_MAX_CH);

  return 0;
}

void nc_draw_gl_npu(struct viewport viewport, int network_task,
                    pp_result_buf *net_result,
                    struct gl_npu_program g_npu_prog) {
  stCnnPostprocessingResults *det_result = &net_result->cnn_result;
  stObjDrawInfo *draw_cnn = &net_result->draw_info;
  stSegDrawInfo *draw_seg = &net_result->seg_info;
  stLaneDrawInfo *draw_lane = &net_result->lane_draw_info;

  int max_class_cnt = draw_cnn->max_class_cnt;
  int max_seg_class_cnt = draw_seg->max_class_cnt;
  float target_view_ratio = 1.f;
  char buftext[128];

  target_view_ratio = (float)WINDOW_HEIGHT / (float)viewport.height;

  glViewport(viewport.x, viewport.y, viewport.width, viewport.height);

  if (network_task == DETECTION) {
    float **color = (float **)malloc(max_class_cnt * sizeof(float *));
    for (int i = 0; i < max_class_cnt; ++i) {
      color[i] = (float *)malloc(RGBA_CNT * sizeof(float));
    }

    for (int j = 0; j < max_class_cnt; ++j) {
      color[j][RGBA_R] = (float)(draw_cnn->class_colors[j].r) / 255.0f;
      color[j][RGBA_G] = (float)(draw_cnn->class_colors[j].g) / 255.0f;
      color[j][RGBA_B] = (float)(draw_cnn->class_colors[j].b) / 255.0f;
      color[j][RGBA_A] = 1.0f;
    }

    for (int i = 0; i < draw_cnn->max_class_cnt; i++) {
      for (int bidx = 0; bidx < det_result->class_objs[i].obj_cnt; bidx++) {
        stObjInfo obj_info = det_result->class_objs[i].objs[bidx];
        nc_opengl_draw_rectangle(obj_info.bbox.x, obj_info.bbox.y,
                                 obj_info.bbox.w, obj_info.bbox.h, color[i],
                                 g_npu_prog);

        // draw bbox label
#ifdef USE_BYTETRACK
        // show track id (not cnn probability)
        if (obj_info.track_id < 0)
          sprintf(buftext, "%s:%0.2f", draw_cnn->class_names[i], obj_info.prob);
        else
          sprintf(buftext, "[%d]%s:%0.2f", obj_info.track_id,
                  draw_cnn->class_names[i], obj_info.prob);
#else
        sprintf(buftext, "%s:%0.2f", draw_cnn->class_names[i], obj_info.prob);
#endif
        float textcolor[3] = {color[i][RGBA_R], color[i][RGBA_G],
                              color[i][RGBA_B]};
        nc_opengl_draw_text(&font_24, buftext, obj_info.bbox.x,
                            (float)WINDOW_HEIGHT - (obj_info.bbox.y - 12),
                            target_view_ratio, textcolor, WINDOW_WIDTH,
                            WINDOW_HEIGHT, g_font_prog);
      }
    }

    for (int i = 0; i < max_class_cnt; ++i) {
      free(color[i]);
    }
    free(color);
  } else if (network_task == SEGMENTATION) {
    glBindTexture(GL_TEXTURE_2D, g_seg_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, draw_seg->width,
                 draw_seg->height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                 det_result->seg);

    float color[COLOR_PALETTE_CNT][RGBA_CNT];

    for (int j = 0; j < max_seg_class_cnt; ++j) {
      color[j][RGBA_R] = (float)(draw_seg->class_colors[j].r) / 255.0f;
      color[j][RGBA_G] = (float)(draw_seg->class_colors[j].g) / 255.0f;
      color[j][RGBA_B] = (float)(draw_seg->class_colors[j].b) / 255.0f;
      color[j][RGBA_A] = (float)(draw_seg->class_colors[j].a) / 255.0f;
    }
    nc_opengl_draw_segmentation(g_seg_texture, (float **)color,
                                draw_seg->max_class_cnt, g_npu_prog);
  } else if (network_task == LANE) {
    float color[COLOR_PALETTE_CNT][RGBA_CNT];

    for (int j = 0; j < draw_lane->max_lane_num; ++j) {
      color[j][RGBA_R] = (float)(draw_lane->index_colors[j].r) / 255.0f;
      color[j][RGBA_G] = (float)(draw_lane->index_colors[j].g) / 255.0f;
      color[j][RGBA_B] = (float)(draw_lane->index_colors[j].b) / 255.0f;
      color[j][RGBA_A] = 1.0f;
    }

    for (int i = 0; i < draw_lane->max_lane_num; i++) {
      int point_num = det_result->lane_det[i].point_cnt;
      int lane_class = det_result->lane_det[i].lane_class;

      if (point_num == 0)
        continue;

      for (int j = 0; j < point_num - 1; j++) {
        float st_x = det_result->lane_det[i].point[j].x;
        float st_y = det_result->lane_det[i].point[j].y;
        float end_x = det_result->lane_det[i].point[j + 1].x;
        float end_y = det_result->lane_det[i].point[j + 1].y;

        nc_opengl_draw_line(st_x, st_y, end_x, end_y, lane_class, color[i],
                            g_npu_prog);
      }
    }

#ifdef USE_UFLD_NETWORK_DEBUGGING
    float color_table[6][RGBA_CNT] = {
        {1.0f, 1.0f, 1.0f, 0.15f}, // white (grid)
        {1.0f, 0.0f, 0.0f, 1.0f},  // red (index 0)
        {0.0f, 0.0f, 1.0f, 1.0f},  // blue (index 1)
        {1.0f, 1.0f, 0.0f, 1.0f},  // yellow (index 2)
        {0.0f, 1.0f, 0.0f, 1.0f},  // green (index 3)
        {0.0f, 0.0f, 0.0f, 1.0f}}; // black (final point)
    stUFLD_dbg_info *debug_info = NULL;
    int info_cnt = 0;

#if defined(DBG_ROW_ANCHOR)

    float row_cell_num = (float)(draw_lane->row_cell_num);
    float row_cell_width = WINDOW_WIDTH / (float)(row_cell_num - 1);
    float row_cell_height =
        (draw_lane->row_anchor_max - draw_lane->row_anchor_min) *
        WINDOW_HEIGHT * (1 / (float)(draw_lane->row_anchor_num - 1));

    // draw row anchor grid
    for (int i = 0; i < (draw_lane->row_anchor_num); i++) {
      nc_opengl_draw_debugging_grid_line(
          0, draw_lane->row_anchor[i] * WINDOW_HEIGHT, WINDOW_WIDTH - 1,
          draw_lane->row_anchor[i] * WINDOW_HEIGHT, 1, color_table[0],
          g_npu_prog);
    }

    for (float j = 0; j < row_cell_num; j++) {
      nc_opengl_draw_debugging_grid_line(
          j / (row_cell_num - 1) * WINDOW_WIDTH, 0,
          j / (row_cell_num - 1) * WINDOW_WIDTH, WINDOW_HEIGHT - 1, 1,
          color_table[0], g_npu_prog);
    }

    for (int i = 0; i < draw_lane->max_lane_num; i++) {
      if (draw_lane->lane_anchor_info[i] != 0)
        continue;

      info_cnt = det_result->lane_det[i].dbg_info_cnt;
      if (info_cnt == 0)
        continue;

      for (int j = 0; j < info_cnt; j++) {
        debug_info = &(det_result->lane_det[i].ufld_dbg_info[j]);

        // draw max_indices
        nc_opengl_draw_rectangle(debug_info->x, debug_info->y, row_cell_width,
                                 row_cell_height, color_table[i + 1],
                                 g_npu_prog);

        // draw stddev
        nc_opengl_draw_line(
            debug_info->index_min, debug_info->y + row_cell_height / 2,
            debug_info->index_max, debug_info->y + row_cell_height / 2, 1,
            color_table[i + 1], g_npu_prog);

        // draw final result (point)
        nc_opengl_draw_rectangle(debug_info->final_point.x - 3,
                                 debug_info->final_point.y - 3, 6, 6,
                                 color_table[5], g_npu_prog);
      }
    }

#elif defined(DBG_COL_ANCHOR)

    float col_cell_num = (float)(draw_lane->col_cell_num);
    float col_cell_width =
        (draw_lane->col_anchor_max - draw_lane->col_anchor_min) * WINDOW_WIDTH *
        (1 / (float)(draw_lane->col_anchor_num - 1));
    float col_cell_height = WINDOW_HEIGHT / (float)(col_cell_num - 1);

    // draw column anchor grid
    for (int i = 0; i < (draw_lane->col_anchor_num); i++) {
      nc_opengl_draw_debugging_grid_line(
          draw_lane->col_anchor[i] * WINDOW_WIDTH, 0,
          draw_lane->col_anchor[i] * WINDOW_WIDTH, WINDOW_HEIGHT - 1, 1,
          color_table[0], g_npu_prog);
    }

    for (float j = 0; j < col_cell_num; j++) {
      nc_opengl_draw_debugging_grid_line(
          0, j / (col_cell_num - 1) * WINDOW_HEIGHT, WINDOW_WIDTH - 1,
          j / (col_cell_num - 1) * WINDOW_HEIGHT, 1, color_table[0],
          g_npu_prog);
    }

    for (int i = 0; i < draw_lane->max_lane_num; i++) {
      if (draw_lane->lane_anchor_info[i] != 1)
        continue;

      info_cnt = det_result->lane_det[i].dbg_info_cnt;
      if (info_cnt == 0)
        continue;

      for (int j = 0; j < info_cnt; j++) {
        debug_info = &(det_result->lane_det[i].ufld_dbg_info[j]);

        // draw max_indices
        nc_opengl_draw_rectangle(debug_info->x, debug_info->y, col_cell_width,
                                 col_cell_height, color_table[i + 1],
                                 g_npu_prog);

        // draw stddev
        nc_opengl_draw_line(
            debug_info->x + col_cell_width / 2, debug_info->index_min,
            debug_info->x + col_cell_width / 2, debug_info->index_max, 1,
            color_table[i + 1], g_npu_prog);

        // draw final result (point)
        nc_opengl_draw_rectangle(debug_info->final_point.x - 3,
                                 debug_info->final_point.y - 3, 6, 6,
                                 color_table[5], g_npu_prog);
      }
    }
#endif
#endif
  } else {
    // printf("Invalid network %d\n", sel_network);
  }
}

void gl_initialize(struct window *window) {
  int width, height, channels;
  char buf[128];

  sprintf(buf, "misc/image/nextchip_s.png");
  image_data = SOIL_load_image(buf, &width, &height, &channels, SOIL_LOAD_RGBA);

  // init video shader
  nc_opengl_init_video_shader(window, 0);

  // init texture for video draw
  for (int i = 0; i < VIDEO_MAX_CH; i++) {
    glGenTextures(1, &window->gl.texture[i]);
    glBindTexture(GL_TEXTURE_2D, window->gl.texture[i]);

    // stbi_set_flip_vertically_on_load(true);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, image_data);
    // glGenerateMipmap(GL_TEXTURE_2D);

    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }

  // Load the font using the freetype library.
  nc_opengl_load_font("misc/font/NotoSans-Regular.ttf", 38, &font_38);
  nc_opengl_load_font("misc/font/NotoSans-Regular.ttf", 24, &font_24);
  // init font shader
  nc_opengl_init_font_shader(&g_font_prog);

  // init npu shader
  nc_opengl_init_npu_shader(&g_npu_prog);

  // init texture for segmentation draw
  glGenTextures(1, &g_seg_texture);
  glBindTexture(GL_TEXTURE_2D, g_seg_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, 0);

  // set viewport
  glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
}

// #define SHOW_FPS
#ifdef SHOW_FPS
float calc_fps_at_loop_ent1(int update_period_fcnt) {
  static uint64_t s_time = 0;
  static float fps = 0.f;
  static uint64_t fcnt = 0;
  uint64_t elapsed_ms = 0;
  fcnt++;
  if (fcnt % update_period_fcnt == 1) {
    if (s_time == 0) {
      s_time = nc_get_mono_time();
    } else {
      elapsed_ms = nc_elapsed_time(s_time);
      // printf("fcnt(%d) elapsed_ms(%d)\n", fcnt, elapsed_ms);
      if (elapsed_ms > 0)
        fps = (float)((fcnt - 1) / (elapsed_ms / 1000.f));

      // re-init
      fcnt = 1;
      s_time = nc_get_mono_time();
    }
  }

  return fps;
}
#endif

/* ============================================================================
 * Render Sub-Functions
 * ============================================================================
 */

/**
 * @brief 카메라 프레임을 캡처하고 OpenGL 텍스처로 업로드
 * @param window Wayland 윈도우 구조체
 * @param networkOrder 네트워크 순서 배열
 */
void render_camera_frames(struct window *window, int *networkOrder) {
  static uint64_t framecnt = 0;

  for (int i = 0; i < VIDEO_MAX_CH; i++) {
    // Mode-based camera selection:
    // DRIVING_MODE: Show both cameras (split screen)
    // PARKING_MODE: Show only CH1 (rear camera, full screen)
    if (g_current_mode == PARKING_MODE && i != 1) {
      continue; // In parking mode, skip CH0
    }

    if (v4l2_config[i].video_buf.video_fd == -1) {
      // Upload texture from logo image data
      glBindTexture(GL_TEXTURE_2D, window->gl.texture[i]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, VIDEO_WIDTH, VIDEO_HEIGHT, 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, image_data);

    } else {
      struct v4l2_buffer video_buf;
      CLEAR(video_buf);

      // Queue the buffer for capturing a new frame
      if (nc_v4l2_dequeue_buffer(v4l2_config[i].video_buf.video_fd,
                                 &video_buf) == -1) {
        // printf("Error VIDIOC_DQBUF buffer %d\n", video_buf.index);
      } else {
#if (VIDEO_MAX_CH == 4)
        if (((framecnt % 2 == 0) && (i == 0 || i == 2)) ||
            ((framecnt % 2 == 1) && (i == 1 || i == 3)))
#endif
        {
          uint64_t time_stamp_us = 0;
          networkOrder[i] = nc_get_cnn_networks_id();
          unsigned char *rgbdata_for_cnn =
              (unsigned char *)malloc(npu_input_info.rgb_size);
#ifdef USE_8MP_VI
          memcpy(dsr_info.dsr_in_buf[0],
                 (uint8_t *)v4l2_config[i]
                     .video_buf.buffers[video_buf.index]
                     .start,
                 MAX_WIDTH_FOR_VDMA_CNN_DS * MAX_HEIGHT_FOR_VDMA_CNN_DS *
                     RGB_CNT);
          dsr_downscale(dsr_fd, &dsr_info, dsr_input_config, dsr_output_config,
                        dsr_config, 0);
          nc_rgb_interleaved_to_planar_neon(
              (unsigned char *)dsr_info.dsr_out_buf[0],
              (unsigned char *)rgbdata_for_cnn,
              (unsigned char *)rgbdata_for_cnn +
                  NPU_INPUT_WIDTH * NPU_INPUT_HEIGHT,
              (unsigned char *)rgbdata_for_cnn +
                  NPU_INPUT_WIDTH * NPU_INPUT_HEIGHT * 2,
              NPU_INPUT_WIDTH, NPU_INPUT_HEIGHT);
#else
          memcpy(rgbdata_for_cnn,
                 (unsigned char *)v4l2_config[i]
                     .video_buf.buffers[video_buf.index]
                     .start,
                 npu_input_info.rgb_size); // 600us
#endif
          send_cnn_buf(rgbdata_for_cnn, time_stamp_us, (uint32_t)i,
                       (E_NETWORK_UID)networkOrder[i]); // 50us
          // printf("send cnn msg : %llu us\n", nc_elapsed_us_time(start_time));
        }
#ifndef USE_8MP_VI
        unsigned char *interleaved_rgb =
            (unsigned char *)malloc(NPU_INPUT_WIDTH * NPU_INPUT_HEIGHT * 3);
        // start_time = nc_get_mono_us_time();
        nc_rgb_planar_to_interleaved_neon(
            (uint8_t *)v4l2_config[i].video_buf.buffers[video_buf.index].start,
            (uint8_t *)v4l2_config[i].video_buf.buffers[video_buf.index].start +
                NPU_INPUT_DATA_SIZE,
            (uint8_t *)v4l2_config[i].video_buf.buffers[video_buf.index].start +
                (NPU_INPUT_DATA_SIZE * 2),
            interleaved_rgb, NPU_INPUT_WIDTH,
            NPU_INPUT_HEIGHT); // NPU ON:3ms, NPU OFF:600us
                               // printf("conv interleaved: %llu us\n",
                               // nc_elapsed_us_time(start_time));
#endif
        if (nc_v4l2_queue_buffer(v4l2_config[i].video_buf.video_fd,
                                 video_buf.index) == -1) {
          printf("Error VIDIOC_QBUF buffer %d\n", video_buf.index);
        }

        // Upload texture from video buffer
        glBindTexture(GL_TEXTURE_2D, window->gl.texture[i]);
#ifdef USE_8MP_VI
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, VIDEO_WIDTH, VIDEO_HEIGHT, 0,
                     GL_RGB, GL_UNSIGNED_BYTE,
                     v4l2_config[i].video_buf.buffers[video_buf.index].start);
#else
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, VIDEO_WIDTH, VIDEO_HEIGHT, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, interleaved_rgb);
        if (interleaved_rgb) {
          // printf("free inter\n");
          free(interleaved_rgb);
        }
#endif
#ifdef SHOW_FPS
        printf(" -- fps(%0.1f)\n", calc_fps_at_loop_ent1(20));
#endif
      }
    }

    // draw video texture
    // Set viewport dynamically based on mode
    if (g_current_mode == PARKING_MODE) {
      // Parking mode: Full screen for rear camera (CH1)
      glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    } else {
      // Driving mode: Split screen (use configured viewport)
      glViewport(g_viewport[i].x, g_viewport[i].y, g_viewport[i].width,
                 g_viewport[i].height);
    }
    nc_opengl_draw_texture(window->gl.texture[i], window);
  }

  framecnt++;

#ifdef USE_ADAS_LD
  draw_ld_output_opengl();
#endif
}

/**
 * @brief NPU 객체 탐지 결과를 렌더링하고 충돌 경고 표시
 * @param window Wayland 윈도우 구조체
 */
void render_detection_results(struct window *window) {
  // draw cnn network result
  for (uint32_t ch = 0; ch < VIDEO_MAX_CH; ch++) {
    // Skip cameras based on mode (same logic as camera rendering)
    if (g_current_mode == PARKING_MODE && ch != 1) {
      continue; // In parking mode, skip CH0
    }

    if (v4l2_config[ch].video_buf.video_fd == -1) {

    } else {
      uint64_t time_stamp = 0;

#ifdef DETECT_NETWORK
      pp_result_buf *det_buf = NULL;
      det_buf = (pp_result_buf *)nc_tsfs_ff_get_readable_buffer_and_timestamp(
          ch + DETECT_NETWORK, &time_stamp);
      if (det_buf) {
        // Use full screen viewport for parking mode, split for driving
        if (g_current_mode == PARKING_MODE) {
          struct viewport full_viewport = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
          nc_draw_gl_npu(full_viewport, det_buf->net_task, det_buf, g_npu_prog);
        } else {
          nc_draw_gl_npu(g_viewport[ch], det_buf->net_task, det_buf,
                         g_npu_prog);
        }

        // Fast approach collision detection
        stObjInfo largest_obj;
        float max_area;
        int obj_class_id = -1;
        if (get_largest_bbox(&det_buf->cnn_result, &det_buf->draw_info,
                             &largest_obj, &max_area, &obj_class_id)) {
          printf("[DETECT] CH%d Class:%d Area:%.0f Y:%.0f\n", ch, obj_class_id,
                 max_area, largest_obj.bbox.y + largest_obj.bbox.h);

          // Y-position + area based immediate danger detection
          float obj_bottom_y = largest_obj.bbox.y + largest_obj.bbox.h;
          bool immediate_danger =
              (obj_bottom_y > 700.0f && max_area > 120000.0f);

          if (immediate_danger || is_fast_approach(ch, max_area, time_stamp)) {
            printf("[ALERT] CH%d Class:%d Area:%.0f Y:%.0f\n", ch, obj_class_id,
                   max_area, obj_bottom_y);
            // Draw thick red warning box
            draw_thick_red_bbox(&largest_obj, g_npu_prog);

            // Add warning text based on mode
            char warn_msg[64];
            if (g_current_mode == DRIVING_MODE) {
              if (ch == FRONT_CH) {
                snprintf(warn_msg, sizeof(warn_msg),
                         "FRONT COLLISION WARNING!");
              } else {
                snprintf(warn_msg, sizeof(warn_msg), "REAR ALERT!");
              }
            } else {
              snprintf(warn_msg, sizeof(warn_msg), "FAST APPROACH!");
            }

            float red[3] = {1.0f, 0.0f, 0.0f};
            int text_x = (g_current_mode == PARKING_MODE)
                             ? WINDOW_WIDTH / 2 - 200
                             : g_viewport[ch].x + 50;
            int text_y = (g_current_mode == PARKING_MODE)
                             ? WINDOW_HEIGHT - 150
                             : g_viewport[ch].height - 50;

            nc_opengl_draw_text(&font_38, warn_msg, text_x, text_y, 1.2f, red,
                                WINDOW_WIDTH, WINDOW_HEIGHT, g_font_prog);
          }
        }
      }
      nc_tsfs_ff_finish_read_buf(ch + DETECT_NETWORK);
#endif

#ifdef SEGMENT_NETWORK
      pp_result_buf *seg_buf = NULL;
      seg_buf = (pp_result_buf *)nc_tsfs_ff_get_readable_buffer_and_timestamp(
          ch + SEGMENT_NETWORK, &time_stamp);
      if (seg_buf) {
        if (g_current_mode == PARKING_MODE) {
          struct viewport full_viewport = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
          nc_draw_gl_npu(full_viewport, seg_buf->net_task, seg_buf, g_npu_prog);
        } else {
          nc_draw_gl_npu(g_viewport[ch], seg_buf->net_task, seg_buf,
                         g_npu_prog);
        }
      }
      nc_tsfs_ff_finish_read_buf(ch + SEGMENT_NETWORK);
#endif

#ifdef LANE_NETWORK
      pp_result_buf *lane_buf = NULL;
      lane_buf = (pp_result_buf *)nc_tsfs_ff_get_readable_buffer_and_timestamp(
          ch + LANE_NETWORK, &time_stamp);
      if (lane_buf) {
        if (g_current_mode == PARKING_MODE) {
          struct viewport full_viewport = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
          nc_draw_gl_npu(full_viewport, lane_buf->net_task, lane_buf,
                         g_npu_prog);
        } else {
          nc_draw_gl_npu(g_viewport[ch], lane_buf->net_task, lane_buf,
                         g_npu_prog);
        }
      }
      nc_tsfs_ff_finish_read_buf(ch + LANE_NETWORK);
#endif
    }
  }
}

/**
 * @brief 주차 모드에서 가이드라인 및 충돌 경고 렌더링
 */
void render_parking_guidelines(void) {
  bool warning_active = false;

  // Define trapezoid coordinates
  float top_y = WINDOW_HEIGHT * 0.98f;
  float bottom_y = WINDOW_HEIGHT * 0.30f;
  float top_left_x = WINDOW_WIDTH * 0.15f;
  float top_right_x = WINDOW_WIDTH * 0.85f;
  float bottom_left_x = WINDOW_WIDTH * 0.40f;
  float bottom_right_x = WINDOW_WIDTH * 0.60f;

  // Collision detection for CH1 (rear camera)
  uint32_t ch = 1; // Rear camera only in parking mode
  if (v4l2_config[ch].video_buf.video_fd != -1) {
    uint64_t time_stamp = 0;

#ifdef DETECT_NETWORK
    pp_result_buf *det_buf = NULL;
    det_buf = (pp_result_buf *)nc_tsfs_ff_get_readable_buffer_and_timestamp(
        ch + DETECT_NETWORK, &time_stamp);
    if (det_buf) {
      // 바운딩 박스 렌더링 (전체 화면)
      struct viewport full_viewport = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
      nc_draw_gl_npu(full_viewport, det_buf->net_task, det_buf, g_npu_prog);

      stCnnPostprocessingResults *det_result = &det_buf->cnn_result;
      stObjDrawInfo *draw_cnn = &det_buf->draw_info;

      // Check all detected objects
      for (int i = 0; i < draw_cnn->max_class_cnt; i++) {
        for (int bidx = 0; bidx < det_result->class_objs[i].obj_cnt; bidx++) {
          stObjInfo obj_info = det_result->class_objs[i].objs[bidx];

          // Calculate object bottom-center point
          float obj_center_x = obj_info.bbox.x + (obj_info.bbox.w / 2.0f);
          float obj_bottom_y = obj_info.bbox.y + obj_info.bbox.h;

          // Check if object is inside trapezoid
          if (is_point_in_trapezoid(obj_center_x, obj_bottom_y, bottom_left_x,
                                    bottom_y, bottom_right_x, bottom_y,
                                    top_right_x, top_y, top_left_x, top_y)) {
            warning_active = true;
            break;
          }
        }
        if (warning_active)
          break;
      }
    }
    nc_tsfs_ff_finish_read_buf(ch + DETECT_NETWORK);
#endif
  }

  // Set guideline color based on warning status
  glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
  float *guide_color;
  float color_normal[4] = {0.0f, 1.0f, 0.0f, 1.0f};  // Green
  float color_warning[4] = {1.0f, 0.0f, 0.0f, 1.0f}; // Red

  if (warning_active) {
    guide_color = color_warning;
  } else {
    guide_color = color_normal;
  }

  // Draw trapezoid outline
  nc_opengl_draw_line(top_left_x, top_y, top_right_x, top_y, 6, guide_color,
                      g_npu_prog);
  nc_opengl_draw_line(top_right_x, top_y, bottom_right_x, bottom_y, 6,
                      guide_color, g_npu_prog);
  nc_opengl_draw_line(bottom_right_x, bottom_y, bottom_left_x, bottom_y, 6,
                      guide_color, g_npu_prog);
  nc_opengl_draw_line(bottom_left_x, bottom_y, top_left_x, top_y, 6,
                      guide_color, g_npu_prog);

  // Draw 3-section dividing lines
  float t1 = 1.0f / 3.0f;
  float t2 = 2.0f / 3.0f;

  float y_mid1 = top_y + t1 * (bottom_y - top_y);
  float x_mid1_left = top_left_x + t1 * (bottom_left_x - top_left_x);
  float x_mid1_right = top_right_x + t1 * (bottom_right_x - top_right_x);
  nc_opengl_draw_line(x_mid1_left, y_mid1, x_mid1_right, y_mid1, 4, guide_color,
                      g_npu_prog);

  float y_mid2 = top_y + t2 * (bottom_y - top_y);
  float x_mid2_left = top_left_x + t2 * (bottom_left_x - top_left_x);
  float x_mid2_right = top_right_x + t2 * (bottom_right_x - top_right_x);
  nc_opengl_draw_line(x_mid2_left, y_mid2, x_mid2_right, y_mid2, 4, guide_color,
                      g_npu_prog);

  // Display WARNING text if active
  if (warning_active) {
    char warn_text[] = "WARNING!";
    float warn_color[3] = {1.0f, 0.0f, 0.0f}; // Red text
    nc_opengl_draw_text(&font_38, warn_text, WINDOW_WIDTH / 2 - 100,
                        WINDOW_HEIGHT / 2, 1.0f, warn_color, WINDOW_WIDTH,
                        WINDOW_HEIGHT, g_font_prog);
  }
}

/**
 * @brief UI 오버레이(FPS 등) 렌더링
 * @param buftext FPS 및 프레임 정보 문자열
 */
void render_ui_overlays(const char *buftext) {
  glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
  float textcolor[3] = {1.0, 0.0, 0.0}; // R, G, B
  nc_opengl_draw_text(&font_38, buftext, 10, 1020, 1.0f, textcolor,
                      WINDOW_WIDTH, WINDOW_HEIGHT, g_font_prog);
}

void render(void *data, struct wl_callback *callback, uint32_t time) {
  static struct timespec begin, end, set_time;
  static uint64_t fpstime = 0;
  static uint64_t fpscount = 0;
  static uint64_t fpscount_00 = 0;
  static uint64_t frametime = 0;
  static uint64_t opengl_time = 0;
  static uint64_t framecnt = 0;
  int networkOrder[VIDEO_MAX_CH];
  static bool s_danger_detected = false; // 위험 감지 상태 추적

  (void)time;

  clock_gettime(CLOCK_MONOTONIC, &begin);
  struct window *window = (struct window *)data;
  // uint64_t start_time = 0;

  assert(window->callback == callback);
  window->callback = NULL;

  if (callback)
    wl_callback_destroy(callback);

  if (!window->configured)
    return;

  // 프레임 시작 시 위험 상태 초기화
  s_danger_detected = false;

  // ============================================
  // Clear screen
  // ============================================
  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // ============================================
  // Render camera frames
  // ============================================
  // render_camera_frames(window, networkOrder); // This function is not defined
  // in the provided context, assuming it's a placeholder for the loop below.
  for (int i = 0; i < VIDEO_MAX_CH; i++) {
    // Mode-based camera selection:
    // DRIVING_MODE: Show both cameras (split screen)
    // PARKING_MODE: Show only CH1 (rear camera, full screen)
    if (g_current_mode == PARKING_MODE && i != 1) {
      continue; // In parking mode, skip CH0
    }

    if (v4l2_config[i].video_buf.video_fd == -1) {
      // Upload texture from logo image data
      glBindTexture(GL_TEXTURE_2D, window->gl.texture[i]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, VIDEO_WIDTH, VIDEO_HEIGHT, 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, image_data);

    } else {
      struct v4l2_buffer video_buf;
      CLEAR(video_buf);

      // Queue the buffer for capturing a new frame
      if (nc_v4l2_dequeue_buffer(v4l2_config[i].video_buf.video_fd,
                                 &video_buf) == -1) {
        // printf("Error VIDIOC_DQBUF buffer %d\n", video_buf.index);
      } else {
#if (VIDEO_MAX_CH == 4)
        if (((framecnt % 2 == 0) && (i == 0 || i == 2)) ||
            ((framecnt % 2 == 1) && (i == 1 || i == 3)))
#endif
        {
          uint64_t time_stamp_us = 0;
          networkOrder[i] = nc_get_cnn_networks_id();
          unsigned char *rgbdata_for_cnn =
              (unsigned char *)malloc(npu_input_info.rgb_size);
#ifdef USE_8MP_VI
          memcpy(dsr_info.dsr_in_buf[0],
                 (uint8_t *)v4l2_config[i]
                     .video_buf.buffers[video_buf.index]
                     .start,
                 MAX_WIDTH_FOR_VDMA_CNN_DS * MAX_HEIGHT_FOR_VDMA_CNN_DS *
                     RGB_CNT);
          dsr_downscale(dsr_fd, &dsr_info, dsr_input_config, dsr_output_config,
                        dsr_config, 0);
          nc_rgb_interleaved_to_planar_neon(
              (unsigned char *)dsr_info.dsr_out_buf[0],
              (unsigned char *)rgbdata_for_cnn,
              (unsigned char *)rgbdata_for_cnn +
                  NPU_INPUT_WIDTH * NPU_INPUT_HEIGHT,
              (unsigned char *)rgbdata_for_cnn +
                  NPU_INPUT_WIDTH * NPU_INPUT_HEIGHT * 2,
              NPU_INPUT_WIDTH, NPU_INPUT_HEIGHT);
#else
          memcpy(rgbdata_for_cnn,
                 (unsigned char *)v4l2_config[i]
                     .video_buf.buffers[video_buf.index]
                     .start,
                 npu_input_info.rgb_size); // 600us
#endif
          send_cnn_buf(rgbdata_for_cnn, time_stamp_us, (uint32_t)i,
                       (E_NETWORK_UID)networkOrder[i]); // 50us
          // printf("send cnn msg : %llu us\n", nc_elapsed_us_time(start_time));
        }
#ifndef USE_8MP_VI
        unsigned char *interleaved_rgb =
            (unsigned char *)malloc(NPU_INPUT_WIDTH * NPU_INPUT_HEIGHT * 3);
        // start_time = nc_get_mono_us_time();
        nc_rgb_planar_to_interleaved_neon(
            (uint8_t *)v4l2_config[i].video_buf.buffers[video_buf.index].start,
            (uint8_t *)v4l2_config[i].video_buf.buffers[video_buf.index].start +
                NPU_INPUT_DATA_SIZE,
            (uint8_t *)v4l2_config[i].video_buf.buffers[video_buf.index].start +
                (NPU_INPUT_DATA_SIZE * 2),
            interleaved_rgb, NPU_INPUT_WIDTH,
            NPU_INPUT_HEIGHT); // NPU ON:3ms, NPU OFF:600us
                               // printf("conv interleaved: %llu us\n",
                               // nc_elapsed_us_time(start_time));
#endif
        if (nc_v4l2_queue_buffer(v4l2_config[i].video_buf.video_fd,
                                 video_buf.index) == -1) {
          printf("Error VIDIOC_QBUF buffer %d\n", video_buf.index);
        }

        // Upload texture from video buffer
        glBindTexture(GL_TEXTURE_2D, window->gl.texture[i]);
#ifdef USE_8MP_VI
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, VIDEO_WIDTH, VIDEO_HEIGHT, 0,
                     GL_RGB, GL_UNSIGNED_BYTE,
                     v4l2_config[i].video_buf.buffers[video_buf.index].start);
#else
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, VIDEO_WIDTH, VIDEO_HEIGHT, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, interleaved_rgb);
        if (interleaved_rgb) {
          // printf("free inter\n");
          free(interleaved_rgb);
        }
#endif
#ifdef SHOW_FPS
        printf(" -- fps(%0.1f)\n", calc_fps_at_loop_ent1(20));
#endif
      }
    }

    // draw video texture
    // Set viewport dynamically based on mode
    if (g_current_mode == PARKING_MODE) {
      // Parking mode: Full screen for rear camera (CH1)
      glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    } else {
      // Driving mode: Split screen (use configured viewport)
      glViewport(g_viewport[i].x, g_viewport[i].y, g_viewport[i].width,
                 g_viewport[i].height);
    }
    nc_opengl_draw_texture(window->gl.texture[i], window);
  }

#ifdef USE_ADAS_LD
  draw_ld_output_opengl();
#endif

  // ============================================
  // Render detection results (주행 모드만)
  // ============================================
  // 주행 모드에서만 detection results 렌더링
  // 주차 모드에서는 render_parking_guidelines()가 CH1 처리
  if (g_current_mode == DRIVING_MODE) {
    // draw cnn network result
    for (uint32_t ch = 0; ch < VIDEO_MAX_CH; ch++) {
      if (v4l2_config[ch].video_buf.video_fd == -1) {
        continue;
      }

      uint64_t time_stamp = 0;

#ifdef DETECT_NETWORK
      pp_result_buf *det_buf = NULL;
      det_buf = (pp_result_buf *)nc_tsfs_ff_get_readable_buffer_and_timestamp(
          ch + DETECT_NETWORK, &time_stamp);
      if (det_buf) {
        // Draw bounding boxes
        nc_draw_gl_npu(g_viewport[ch], det_buf->net_task, det_buf, g_npu_prog);

        // Collision detection
        stObjInfo largest_obj;
        float max_area;
        int obj_class_id = -1;
        if (get_largest_bbox(&det_buf->cnn_result, &det_buf->draw_info,
                             &largest_obj, &max_area, &obj_class_id)) {
          printf("[DETECT] CH%d Class:%d Area:%.0f Y:%.0f\n", ch, obj_class_id,
                 max_area, largest_obj.bbox.y + largest_obj.bbox.h);

          bool is_danger = false;
          float obj_bottom_y = largest_obj.bbox.y + largest_obj.bbox.h;

          // 전방/후방 충돌감지 로직 차별화
          if (ch == FRONT_CH) {
            // 전방: 빠른 접근 감지 (객체가 빠르게 가까워지는지)
            bool immediate_danger =
                (obj_bottom_y > 700.0f && max_area > 120000.0f);
            is_danger =
                immediate_danger || is_fast_approach(ch, max_area, time_stamp);
          } else if (ch == REAR_CH) {
            // 후방: 거리 기반 (화면 하단 + 큰 면적)
            // 후방은 전방보다 더 낮은 Y 위치에서 경고 (후방은 가까운 거리에
            // 민감)
            bool close_proximity =
                (obj_bottom_y > 600.0f && max_area > 80000.0f);
            is_danger =
                close_proximity || is_fast_approach(ch, max_area, time_stamp);
          }

          if (is_danger) {
            s_danger_detected = true;
            printf("[ALERT] CH%d Class:%d Area:%.0f Y:%.0f\n", ch, obj_class_id,
                   max_area, obj_bottom_y);
            // Draw thick red warning box
            draw_thick_red_bbox(&largest_obj, g_npu_prog);

            // Warning text
            char warn_msg[64];
            if (ch == FRONT_CH) {
              snprintf(warn_msg, sizeof(warn_msg), "FRONT COLLISION WARNING!");
            } else {
              snprintf(warn_msg, sizeof(warn_msg), "REAR COLLISION WARNING!");
            }

            float red[3] = {1.0f, 0.0f, 0.0f};
            int text_x = g_viewport[ch].x + 50;
            int text_y = g_viewport[ch].height - 50;

            nc_opengl_draw_text(&font_38, warn_msg, text_x, text_y, 1.2f, red,
                                WINDOW_WIDTH, WINDOW_HEIGHT, g_font_prog);
          }
        }
      }
      nc_tsfs_ff_finish_read_buf(ch + DETECT_NETWORK);
#endif
    }
  }
  // if (g_current_mode == DRIVING_MODE) 끝

#ifdef SEGMENT_NETWORK
  // Segment는 주차모드에서만 표시 (주행모드에서는 화면이 복잡해짐)
  if (g_current_mode == PARKING_MODE) {
    uint32_t ch = 1; // 후방 카메라만
    if (v4l2_config[ch].video_buf.video_fd != -1) {
      uint64_t time_stamp = 0;
      pp_result_buf *seg_buf = NULL;
      seg_buf = (pp_result_buf *)nc_tsfs_ff_get_readable_buffer_and_timestamp(
          ch + SEGMENT_NETWORK, &time_stamp);
      if (seg_buf) {
        struct viewport full_viewport = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        nc_draw_gl_npu(full_viewport, seg_buf->net_task, seg_buf, g_npu_prog);
      }
      nc_tsfs_ff_finish_read_buf(ch + SEGMENT_NETWORK);
    }
  }
#endif

#ifdef LANE_NETWORK
  // Lane detection: 주행모드에서 전방 카메라만 표시, 후방 버퍼는 비워줌
  if (g_current_mode == DRIVING_MODE) {
    for (uint32_t ch = 0; ch < VIDEO_MAX_CH; ch++) {
      if (v4l2_config[ch].video_buf.video_fd == -1) {
        continue;
      }

      uint64_t time_stamp = 0;
      pp_result_buf *lane_buf = NULL;
      lane_buf = (pp_result_buf *)nc_tsfs_ff_get_readable_buffer_and_timestamp(
          ch + LANE_NETWORK, &time_stamp);

      // 전방 카메라(CH0)만 렌더링, 후방은 버퍼만 비움
      if (lane_buf && ch == FRONT_CH) {
        printf("[LANE] Rendering CH%d lane detection\n", ch);
        nc_draw_gl_npu(g_viewport[ch], lane_buf->net_task, lane_buf,
                       g_npu_prog);
      } else if (lane_buf) {
        printf("[LANE] Skipping render for CH%d (rear camera)\n", ch);
      }
      nc_tsfs_ff_finish_read_buf(ch + LANE_NETWORK);
    }
  }
#endif

  framecnt++;
  fpscount += 1;
  char buftext[256];

  // 현재 모드 문자열만 표시
  const char *mode_str =
      (g_current_mode == DRIVING_MODE) ? "DRIVING" : "PARKING";

  sprintf(buftext, "[%s]", mode_str);

  // ============================================
  // Render UI overlays (FPS counter)
  // ============================================
  render_ui_overlays(buftext);

  // ============================================
  // Calculate rendering time
  // ============================================
  clock_gettime(CLOCK_MONOTONIC, &end);

  opengl_time = ((end.tv_sec - begin.tv_sec) * 1000 +
                 (end.tv_nsec - begin.tv_nsec) / 1000000);

  frametime = ((end.tv_sec - set_time.tv_sec) * 1000 +
               (end.tv_nsec - set_time.tv_nsec) / 1000000);

  set_time = end;

  fpstime = fpstime + frametime;

  if (fpstime >= 1000) {
    if (fpscount_00 > 0) {
      fpscount_00 = (fpscount_00 + fpscount) / 2;
    } else {
      fpscount_00 = fpscount;
    }

    fpscount = 0;
    fpstime = 0;
  }

  // ============================================
  // PARKING MODE GUIDELINE DRAWING
  // ============================================
  if (g_current_mode == PARKING_MODE) {
    render_parking_guidelines();
  }

  // ============================================
  // Update Wayland display
  // ============================================
  nc_wayland_display_draw(window, (void *)render);
}

int npu_init(st_npu_input_info *npu_input_info) {
  /* Initialize CNN */
  if (nc_aiw_init_cnn() < 0) {
    fprintf(stderr, "nc_aiw_init_cnn() failure!!\n");
    return -1;
  }
#ifdef SHOW_YOLOV8_DETECT
  if (nc_aiw_add_network_to_builder(
          nc_localize_path((const char *)NETWORK_FILE_YOLOV8_DET),
          NETWORK_YOLOV8_DET, nc_postprocess_yolov8_inference_result) < 0) {
    fprintf(stderr, "nc_aiw_add_network_to_builder() failure!!\n");
    return -1;
  }
#endif
#ifdef SHOW_YOLOV5_DETECT
  if (nc_aiw_add_network_to_builder(
          nc_localize_path((const char *)NETWORK_FILE_YOLOV5_DET),
          NETWORK_YOLOV5_DET, nc_postprocess_yolov5_inference_result) < 0) {
    fprintf(stderr, "nc_aiw_add_network_to_builder() failure!!\n");
    return -1;
  }
#endif
#ifdef SHOW_PELEE_SEG
  if (nc_aiw_add_network_to_builder(
          nc_localize_path((const char *)NETWORK_FILE_PELEE_SEG),
          NETWORK_PELEE_SEG,
          nc_postprocess_segmentation_inference_result) < 0) {
    fprintf(stderr, "nc_aiw_add_network_to_builder() failure!!\n");
    return -1;
  }
#endif
#ifdef SHOW_PELEE_DETECT
  if (nc_aiw_add_network_to_builder(
          nc_localize_path((const char *)NETWORK_FILE_PELEE_DET),
          NETWORK_PELEE_DET, nc_postprocess_pelee_inference_result) < 0) {
    fprintf(stderr, "nc_aiw_add_network_to_builder() failure!!\n");
    return -1;
  }
#endif
#ifdef SHOW_UFLD_LANE
  if (nc_aiw_add_network_to_builder(
          nc_localize_path((const char *)NETWORK_FILE_UFLD_LANE),
          NETWORK_UFLD_LANE, nc_postprocess_ufld_inference_result) < 0) {
    fprintf(stderr, "nc_aiw_add_network_to_builder() failure!!\n");
    return -1;
  }
#endif
#ifdef SHOW_TRI_CHIMERA
  if (nc_aiw_add_network_to_builder(
          nc_localize_path((const char *)NETWORK_FILE_TRI_CHIMERA),
          NETWORK_TRI_CHIMERA,
          nc_postprocess_trichimera_inference_result) < 0) {
    fprintf(stderr, "nc_aiw_add_network_to_builder() failure!!\n");
    return -1;
  }
#endif
  if (nc_aiw_finish_network_builder() < 0) {
    fprintf(stderr, "nc_aiw_finish_network_builder() failure!!\n");
    return -1;
  }
  // obtain the input resoltution for the CNN network
  // get information of the input tensor
  aiwTensorInfo in_tinfo;
#ifdef SHOW_YOLOV8_DETECT
  if (nc_get_cnn_network_input_resol(NETWORK_YOLOV8_DET, &in_tinfo) < 0) {
    printf("failed to get the input resolution for the CNN network\n");
  }
#endif
#ifdef SHOW_YOLOV5_DETECT
  if (nc_get_cnn_network_input_resol(NETWORK_YOLOV5_DET, &in_tinfo) < 0) {
    printf("failed to get the input resolution for the CNN network\n");
  }
#endif
#ifdef SHOW_PELEE_SEG
  if (nc_get_cnn_network_input_resol(NETWORK_PELEE_SEG, &in_tinfo) < 0) {
    printf("failed to get the input resolution for the CNN network\n");
  }
#endif
#ifdef SHOW_PELEE_DETECT
  if (nc_get_cnn_network_input_resol(NETWORK_PELEE_DET, &in_tinfo) < 0) {
    printf("failed to get the input resolution for the CNN network\n");
  }
#endif
#ifdef SHOW_UFLD_LANE
  if (nc_get_cnn_network_input_resol(NETWORK_UFLD_LANE, &in_tinfo) < 0) {
    printf("failed to get the input resolution for the CNN network\n");
  }
#endif
#ifdef SHOW_TRI_CHIMERA
  if (nc_get_cnn_network_input_resol(NETWORK_TRI_CHIMERA, &in_tinfo) < 0) {
    printf("failed to get the input resolution for the CNN network\n");
  }
#endif

  npu_input_info->w = in_tinfo.dim.w; // network input width
  npu_input_info->h = in_tinfo.dim.h; // network input height
  npu_input_info->rgb_size = in_tinfo.dim.w * in_tinfo.dim.h * RGB_CNT;

#ifdef USE_BYTETRACK
  for (int i = 0; i < VIDEO_MAX_CH; i++) {
    E_NETWORK_UID net_id;
#ifdef SHOW_PELEE_DETECT
    net_id = NETWORK_PELEE_DET;
#endif
#ifdef SHOW_YOLOV5_DETECT
    net_id = NETWORK_YOLOV5_DET;
#endif
#ifdef SHOW_YOLOV8_DETECT
    net_id = NETWORK_YOLOV8_DET;
#endif
#ifndef SHOW_PELEE_SEG
    if (nc_init_bytetrackers(CAP_FPS, i, net_id) != 0) {
      perror("nc_init_bytetrackers() error");
      return -1;
    }
#endif
  }
#endif

  printf("aiw finish\n");

  return 0;
}

// Helper function to check if a point is inside a trapezoid
bool is_point_in_trapezoid(float px, float py, float x1, float y1, float x2,
                           float y2, float x3, float y3, float x4, float y4) {
  // Cross product for edge (x1,y1) -> (x2,y2) with point (px,py)
  float d1 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
  // Cross product for edge (x2,y2) -> (x3,y3) with point (px,py)
  float d2 = (x3 - x2) * (py - y2) - (y3 - y2) * (px - x2);
  // Cross product for edge (x3,y3) -> (x4,y4) with point (px,py)
  float d3 = (x4 - x3) * (py - y3) - (y4 - y3) * (px - x3);
  // Cross product for edge (x4,y4) -> (x1,y1) with point (px,py)
  float d4 = (x1 - x4) * (py - y4) - (y1 - y4) * (px - x4);

  // If all cross products have the same sign, the point is inside
  bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0) || (d4 < 0);
  bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0) || (d4 > 0);

  return !(has_neg && has_pos); // True if all same sign or zero
}

void mode_change_handler(int signo) {
  if (signo == SIGUSR1) {
    g_current_mode = DRIVING_MODE;
    printf("\n[MODE] >>> Switched to DRIVING MODE (Front Camera) <<<\n");
  } else if (signo == SIGUSR2) {
    g_current_mode = PARKING_MODE;
    printf("\n[MODE] >>> Switched to PARKING MODE (Rear Camera) <<<\n");
  }
}

static void signal_int() {
  running = 0;
  nc_cnn_postprocess_stop();
}

static void usage(int error_code) {
  fprintf(stderr, "Usage: simple-egl [OPTIONS]\n\n"
                  "  -f\tRun in fullscreen mode\n"
                  "  -o\tCreate an opaque surface\n"
                  "  -h\tThis help text\n\n");

  exit(error_code);
}

void *cnn_task(void *arg) {
  (void)arg;
  printf("CNN TASK RUN!!\n");

  while (running) {
    stCnnData *cnn_data = NULL;
    if (receive_cnn_buf(&cnn_data) != -1) {
      nc_aiw_run_cnn(cnn_data->ptr_cnn_buf, cnn_data->time_stamp_us,
                     cnn_data->cam_ch, cnn_data->net_id);
      if (cnn_data) {
        free(cnn_data->ptr_cnn_buf);
        free(cnn_data);
      }
    }
  }

  printf("EXIT CNN_TASK!!\n");
  return NULL;
}

/* ========================= COLLISION DETECTION HELPERS
 * ========================= */

/* Find the largest bounding box by area from detection results */
bool get_largest_bbox(const stCnnPostprocessingResults *det_result,
                      const stObjDrawInfo *draw_info, stObjInfo *out_obj,
                      float *out_area, int *out_class_id) {
  if (!det_result || !draw_info || !out_obj || !out_area || !out_class_id)
    return false;

  float best_area = 0.0f;
  bool found = false;
  stObjInfo best = {0};
  int best_class_id = -1;

  for (int cls = 0; cls < draw_info->max_class_cnt; cls++) {
    int cnt = det_result->class_objs[cls].obj_cnt;
    int class_id = det_result->class_objs[cls].class_id;
    for (int k = 0; k < cnt; k++) {
      stObjInfo obj = det_result->class_objs[cls].objs[k];
      float area = obj.bbox.w * obj.bbox.h;
      if (area > best_area) {
        best_area = area;
        best = obj;
        best_class_id = class_id;
        found = true;
      }
    }
  }

  if (!found)
    return false;
  *out_obj = best;
  *out_area = best_area;
  *out_class_id = best_class_id;
  return true;
}

/* Draw thick red bounding box by overlaying multiple rectangles */
void draw_thick_red_bbox(const stObjInfo *obj, struct gl_npu_program prog) {
  if (!obj)
    return;

  float red[RGBA_CNT] = {1.0f, 0.0f, 0.0f, 1.0f};

  // Draw 3 overlapping rectangles for thickness
  for (int t = 0; t < 3; t++) {
    float x = obj->bbox.x - (float)t;
    float y = obj->bbox.y - (float)t;
    float w = obj->bbox.w + (float)(2 * t);
    float h = obj->bbox.h + (float)(2 * t);
    nc_opengl_draw_rectangle(x, y, w, h, red, prog);
  }
}

/* Detect fast approach based on bbox area growth rate */
bool is_fast_approach(uint32_t ch, float cur_area, uint64_t now_us) {
  if (ch >= VIDEO_MAX_CH)
    return false;

  uint64_t prev_us = g_prev_ts_us[ch];
  float prev_a = g_prev_max_area[ch];

  // First frame or initial state
  if (prev_us == 0 || prev_a <= 0.0f) {
    g_prev_ts_us[ch] = now_us;
    g_prev_max_area[ch] = cur_area;
    return false;
  }

  double dt = (double)(now_us - prev_us) / 1000000.0;
  if (dt <= 0.0)
    dt = 1e-6;

  // Always update state
  g_prev_ts_us[ch] = now_us;
  g_prev_max_area[ch] = cur_area;

  // Noise filtering
  if (prev_a < WARN_MIN_PREV_AREA)
    return false;
  if (cur_area < WARN_MIN_AREA)
    return false;

  float factor = cur_area / prev_a;

  // Fast approach: significant growth in short time
  if (dt <= WARN_MAX_DT_SEC && factor >= WARN_GROWTH_FACTOR) {
    return true;
  }

  return false;
}
/* Check if wall/building pixels exceed threshold in guideline area */
bool check_wall_in_guideline(const uint8_t *seg_data, int seg_width,
                             int seg_height, float x1, float y1, float x2,
                             float y2, float x3, float y3, float x4, float y4) {
  if (!seg_data || seg_width <= 0 || seg_height <= 0)
    return false;

  int wall_pixel_count = 0;
  int total_pixels = 0;

  float min_x = fminf(fminf(x1, x2), fminf(x3, x4));
  float max_x = fmaxf(fmaxf(x1, x2), fmaxf(x3, x4));
  float min_y = fminf(fminf(y1, y2), fminf(y3, y4));
  float max_y = fmaxf(fmaxf(y1, y2), fmaxf(y3, y4));

  for (int y = (int)min_y; y < (int)max_y && y < seg_height; y++) {
    for (int x = (int)min_x; x < (int)max_x && x < seg_width; x++) {
      if (is_point_in_trapezoid((float)x, (float)y, x1, y1, x2, y2, x3, y3, x4,
                                y4)) {
        int idx = y * seg_width + x;
        uint8_t class_id = seg_data[idx];
        total_pixels++;
        if (class_id == SEG_CLASS_BUILDING || class_id == SEG_CLASS_WALL) {
          wall_pixel_count++;
        }
      }
    }
  }

  if (total_pixels == 0)
    return false;
  float wall_ratio = (float)wall_pixel_count / (float)total_pixels;
  if (wall_ratio > 0.1f) {
    printf("[WALL] Ratio=%.2f (%d/%d)\\n", wall_ratio, wall_pixel_count,
           total_pixels);
  }
  return (wall_ratio > SEG_WALL_THRESHOLD);
}

int main(int argc, char **argv) {
  struct sigaction sigint;
  struct display display;
  struct window window;
  int i, ret = 0;
  pthread_t p_thread[MAX_TASK_CNT];
  int task_cnt = 0;
  int thr_id;
  int status;

  for (i = 1; i < argc; i++) {
    if (strcmp("-f", argv[i]) == 0)
      window.fullscreen = 1;
    else if (strcmp("-o", argv[i]) == 0)
      window.opaque = 1;
    else if (strcmp("-h", argv[i]) == 0)
      usage(EXIT_SUCCESS);
    else
      usage(EXIT_FAILURE);
  }

  sigint.sa_handler = (sighandler_t)signal_int;
  sigemptyset(&sigint.sa_mask);
  sigint.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &sigint, NULL);

  // Register mode switching signal handlers
  struct sigaction sa_mode;
  sa_mode.sa_handler = mode_change_handler;
  sigemptyset(&sa_mode.sa_mask);
  sa_mode.sa_flags = 0;

  if (sigaction(SIGUSR1, &sa_mode, NULL) == -1) {
    perror("Error: cannot handle SIGUSR1");
  }
  if (sigaction(SIGUSR2, &sa_mode, NULL) == -1) {
    perror("Error: cannot handle SIGUSR2");
  }
  printf("[INFO] Mode switching enabled: SIGUSR1=Driving, SIGUSR2=Parking\n");

  memset(&display, 0, sizeof(display));
  memset(&window, 0, sizeof(window));
  memset(&v4l2_config, 0, sizeof(v4l2_config));

  set_v4l2_config();

  nc_init_path_localizer();

  window.display = &display;
  display.window = &window;
  window.window_size.width = WINDOW_WIDTH;
  window.window_size.height = WINDOW_HEIGHT;

  set_viewport_config();

  nc_wayland_display_init(&display, (void *)render);

  ret = v4l2_initialize();
  if (ret < 0) {
    printf("Error v4l2_initialize\n");
    return -1;
  }

  gl_initialize(&window);
#ifdef USE_8MP_VI
  dsr_init();
#endif
#ifdef USE_ADAS_LD
  ld_opengl_program_set(&g_npu_prog);
  NC_ADAS_OPEN();
#endif

  // create thread-safe flip-flop buffer
  for (int i = 0; i < VIDEO_MAX_CH; i++) {
    if (v4l2_config[i].video_buf.video_fd == -1) {

    } else {
#ifdef DETECT_NETWORK
      int det_buf_size = sizeof(pp_result_buf);
      if (nc_tsfs_ff_create_buffers(i + DETECT_NETWORK, det_buf_size) < 0) {
        exit(1);
      }
#endif
#ifdef SEGMENT_NETWORK
      int seg_buf_size = sizeof(pp_result_buf);
      if (nc_tsfs_ff_create_buffers(i + SEGMENT_NETWORK, seg_buf_size) < 0) {
        exit(1);
      }
#endif
#ifdef LANE_NETWORK
      int lane_buf_size = sizeof(pp_result_buf);
      if (nc_tsfs_ff_create_buffers(i + LANE_NETWORK, lane_buf_size) < 0) {
        exit(1);
      }
#endif
    }
  }
  mq_unlink(MQ_NAME_CNN_BUF);

  if (npu_init(&npu_input_info) < 0) {
    printf("failed to init NPU\n");
    return -1;
  }

  printf("create tasks\n");
  thr_id = pthread_create(&p_thread[task_cnt++], NULL, cnn_task, (void *)NULL);
  if (thr_id < 0) {
    perror("thread create error : cnn_task");
    exit(1);
  }

  cnn_postprocess_arg cnn_post_param;
  cnn_post_param.target_width = WINDOW_WIDTH;
  cnn_post_param.target_height = WINDOW_HEIGHT;
  thr_id = pthread_create(&p_thread[task_cnt++], NULL, nc_cnn_postprocess_task,
                          (void *)&cnn_post_param);
  if (thr_id < 0) {
    perror("thread create error : nc_cnn_postprocess_task");
    exit(1);
  }

#ifdef USE_ADAS_LD
  thr_id = pthread_create(&p_thread[task_cnt++], NULL, ld_task, NULL);
  if (thr_id < 0) {
    perror("thread create error : ld_task");
    exit(1);
  }
#endif

  while (running && ret != -1) {
    ret = wl_display_dispatch(display.display);
  }

  for (int i = 0; i < task_cnt; i++) {
    pthread_join(p_thread[i], (void **)&status);
  }

#ifdef USE_ADAS_LD
  NC_ADAS_CLOSE();
#endif

#ifdef USE_BYTETRACK
  for (int i = 0; i < VIDEO_MAX_CH; i++) {
    nc_deInit_bytetrackers(i);
  }
#endif

#ifdef USE_8MP_VI
  dsr_deinit();
#endif
  nc_wayland_display_destroy(&display);

  return 0;
}
