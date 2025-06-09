import os
import leafmap
from samgeo import SamGeo, show_image, download_file, overlay_images, tms_to_geotiff
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

m = leafmap.Map(center=[37.8713, -122.2580], zoom=17, height="800px")
m.add_basemap("SATELLITE")

if m.user_roi_bounds() is not None:
    bbox = m.user_roi_bounds()
else:
    bbox = [-122.2659, 37.8682, -122.2521, 37.8741]

image = "chongming_google_subset.tif"

# ## 可视化原始谷歌影像
# # 读取原始图像数据，**不进行任何像素值修改**
# original_data = cv2.imread(image, -1)  # 使用 -1 参数读取原始深度的图像

# # 如果需要，对原始图像数据进行裁剪（例如，只保留前4000x4000像素）
# cropped_original_data = original_data[:4000, :4000, :]

# # 定义存储裁剪后原始图像的文件路径
# output_image_path = "google_chongming_4000_4000.png"  # 可以自定义输出文件名和格式

# # 使用cv2.imwrite()存储裁剪后的原始图像
# # 注意：如果原始图像是浮点数格式，可能需要先转换为uint8类型才能保存为常见的图像格式
# if original_data.dtype == 'float32' or original_data.dtype == 'float64':
#     # 将浮点数数据转换为 [0, 255] 范围的 uint8 类型
#     cv2.imwrite(output_image_path, (cropped_original_data * 255).astype("uint8"))
# else:
#     # 如果已经是 uint8 类型，直接保存
#     cv2.imwrite(output_image_path, cropped_original_data)

# exit()


sam_kwargs = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,
}

# sam_kwargs = {
#     "points_per_side": 32,
#     "pred_iou_thresh": 0.3, #0.86,
#     "stability_score_thresh": 0.8, #0.92,
#     "crop_n_layers": 1,
#     "crop_n_points_downscale_factor": 2,
#     "min_mask_region_area": 50,#100,
# }
checkpoint='/mnt/data/zjq/sam_chongming_label/sam_vit_h_4b8939.pth'
sam = SamGeo(
    model_type="vit_h",
    checkpoint=checkpoint,
    sam_kwargs=sam_kwargs,
)

###
## 00-[:2000,:2000,:]; 01-[2000:4000,:2000,:]; 10-[:2000,2000:4000,:]; 11-[2000:4000,2000:4000,:]
data = (cv2.imread(image,-1)*255).astype("uint8")

# 定义将图像分成4个部分的坐标范围
parts = [
    (slice(None, 2000), slice(None, 2000)),
    (slice(2000, 4000), slice(None, 2000)),
    (slice(None, 2000), slice(2000, 4000)),
    (slice(2000, 4000), slice(2000, 4000))
]

# # 定义将图像分成16个部分的坐标范围
# parts = [
#     (slice(None, 1000), slice(None, 1000)),
#     (slice(1000, 2000), slice(None, 1000)),
#     (slice(None, 1000), slice(1000, 2000)),
#     (slice(1000, 2000), slice(1000, 2000)),
#     (slice(2000, 3000), slice(None, 1000)),
#     (slice(3000, 4000), slice(None, 1000)),
#     (slice(2000, 3000), slice(1000, 2000)),
#     (slice(3000, 4000), slice(1000, 2000)),
#     (slice(None, 1000), slice(2000, 3000)),
#     (slice(1000, 2000), slice(2000, 3000)),
#     (slice(None, 1000), slice(3000, 4000)),
#     (slice(1000, 2000), slice(3000, 4000)),
#     (slice(2000, 3000), slice(2000, 3000)),
#     (slice(3000, 4000), slice(2000, 3000)),
#     (slice(2000, 3000), slice(3000, 4000)),
#     (slice(3000, 4000), slice(3000, 4000))
# ]

# 循环处理每个部分
for i, (row_slice, col_slice) in enumerate(parts):
    # 获取当前部分的数据
    part_data = data[row_slice, col_slice]
  
    # 将十进制索引转换为4位二进制字符串
    file_suffix = ["00","01","10","11"][i] #format(i, '04b')  # '04b' 表示4位二进制，不足4位在左侧填充零
  
    # 生成masks和annotations文件
    sam.generate(image, part_data, output=f"masks_chongming_{file_suffix}.tif", foreground=True, unique=True)
    sam.show_masks(cmap="binary_r")
    sam.show_anns(axis="off", alpha=1, output=f"annotations_chongming_{file_suffix}.tif", blend=False)
  
    print(f"Processed part {file_suffix}")

# print(2)
# sam.generate(image, data, output="masks_chongming.tif", foreground=True, unique=True)#, batch=True)
# print(3)
# sam.show_masks(cmap="binary_r")

# sam.show_anns(axis="off", alpha=1, output="annotations_chongming.tif",blend=False)
# print(4)
