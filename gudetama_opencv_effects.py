import cv2
import numpy as np
import time

def apply_effects_to_video(input_video_path, output_video_path):
    """
    對影片應用一系列OpenCV特效，並在左上角顯示特效名稱。
    小畫面會顯示原始影片，且與主畫面同步播放。

    Args:
        input_video_path (str): 輸入影片的路徑。
        output_video_path (str): 輸出影片的路徑。
    """

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 {input_video_path}")
        return

    # 取得影片的屬性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"影片資訊：FPS={fps}, 寬={width}, 高={height}, 總幀數={total_frames}")

    # 小畫面尺寸（原影片的1/4大小）
    pip_width = width // 4
    pip_height = height // 4
    
    # 小畫面位置（左側中間）
    pip_x = 10  # 距離左邊的距離
    pip_y = (height - pip_height) // 2  # 垂直居中
    
    # 小畫面邊框參數
    border_thickness = 2
    border_color = (255, 255, 255)  # 白色邊框

    # 定義輸出影片的編碼器和檔案名
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或 'XVID' for .avi
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 定義每個特效的時間範圍 (秒) 和對應的英文名稱
    effects_timeline = {
        "original": (0, 9, "1. Original Video"),
        "rotation_scale": (9, 19, "2. Rotation & Scale"),
        "edge_detection": (19, 28, "3. Canny Edge Detection"),
        "morphological_gradient": (28, 36, "4. Morphological Gradient"),
        "flip": (36, 46, "5. Flip - Four Panels Mirror"),
        "colormap": (46, 55, "6. ColorMap"),
        "dog_filter": (55, 64, "7. DoG Filter"),
        "gamma_correction": (64, 74, "8. Gamma Correction"),
        "face_detection_mosaic": (74, 85, "9. Face Detection + Mosaic : 2"),
        "sift_keypoints": (85, 95, "10. SIFT Keypoint Detection")
    }

    # 調整時間範圍為幀數
    effects_frame_ranges = {}
    for effect_name, (start_sec, end_sec, display_name) in effects_timeline.items():
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        effects_frame_ranges[effect_name] = (start_frame, end_frame, display_name)

    # 臉部偵測的 Haar Cascade 分類器
    # 請替換為你實際的 haarcascade_frontalface_default.xml 路徑
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # SIFT 偵測器
    sift = cv2.SIFT_create()

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read() # 讀取原始影片幀
        if not ret:
            break

        current_effect = "original"
        current_display_name = "Original Video"
        for effect_name, (start_frame, end_frame, display_name) in effects_frame_ranges.items():
            if start_frame <= frame_count <= end_frame:
                current_effect = effect_name
                current_display_name = display_name
                break

        processed_frame = frame.copy() # 建立一個副本用於處理，不影響原始幀

        # 應用特效到 processed_frame
        if current_effect == "rotation_scale":
            # 旋轉與縮放
            angle = (frame_count - effects_frame_ranges["rotation_scale"][0]) * 5  # 每幀增加5度
            scale = 1 + (frame_count - effects_frame_ranges["rotation_scale"][0]) * 0.005 # 每幀增加一點縮放

            # 確保縮放不會過大或過小，這裡設定上限
            scale = min(scale, 2.0)

            M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            processed_frame = cv2.warpAffine(frame, M, (width, height))

        elif current_effect == "edge_detection":
            # Sobel, Canny, Laplacian 邊緣偵測
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Canny
            processed_frame = cv2.Canny(gray, 100, 200)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR) # 轉回BGR以便寫入彩色影片

        elif current_effect == "morphological_gradient":
            # 形態學梯度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed_frame = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        elif current_effect == "flip":
            # 四分割畫面，左右鏡像效果
            # 計算每個子畫面的尺寸
            quad_width = width // 2
            quad_height = height // 2
            
            # 創建一個空白畫面用於組合
            processed_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 縮放原始幀到子畫面大小
            resized_frame = cv2.resize(frame, (quad_width, quad_height))
            
            # 創建鏡像版本（水平翻轉）
            mirrored_frame = cv2.flip(resized_frame, 1)
            
            # 創建flip版本（垂直和水平翻轉）
            flipped_frame = cv2.flip(resized_frame, -1)
            
            # 創建flip且鏡像版本
            flipped_mirrored_frame = cv2.flip(flipped_frame, 1)
            
            # 放置四個畫面
            # 左上：原始
            processed_frame[0:quad_height, 0:quad_width] = resized_frame
            
            # 右上：鏡像
            processed_frame[0:quad_height, quad_width:width] = mirrored_frame
            
            # 左下：flip
            processed_frame[quad_height:height, 0:quad_width] = flipped_frame
            
            # 右下：flip且鏡像
            processed_frame[quad_height:height, quad_width:width] = flipped_mirrored_frame
            
            # 添加分割線
            line_color = (255, 255, 255)  # 白色分割線
            line_thickness = 1
            
            # 垂直分割線
            cv2.line(processed_frame, (quad_width, 0), (quad_width, height), line_color, line_thickness)
            
            # 水平分割線
            cv2.line(processed_frame, (0, quad_height), (width, quad_height), line_color, line_thickness)

        elif current_effect == "colormap":
            # 彩色映射
            # 選擇一個彩色映射類型，這裡我用 TURBO
            # 其他選項：cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_RAINBOW, etc.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        elif current_effect == "dog_filter":
            # Dog濾波 (高斯差分)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gaussian1 = cv2.GaussianBlur(gray, (5, 5), 0)
            gaussian2 = cv2.GaussianBlur(gray, (9, 9), 0) # 使用不同的 sigma 值
            processed_frame = gaussian1 - gaussian2
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        elif current_effect == "gamma_correction":
            # Gamma 校正
            gamma = 0.5 + (frame_count - effects_frame_ranges["gamma_correction"][0]) * 0.02 # 從0.5逐漸增加

            # 建立一個查找表
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            processed_frame = cv2.LUT(frame, table)

        elif current_effect == "face_detection_mosaic":
            # 臉部偵測與馬賽克
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                # 提取臉部區域
                face_roi = processed_frame[y:y+h, x:x+w]
                # 對臉部區域進行縮小和放大以創建馬賽克效果
                if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    small_face = cv2.resize(face_roi, (w//8, h//8), interpolation=cv2.INTER_LINEAR)
                    mosaic_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)
                    processed_frame[y:y+h, x:x+w] = mosaic_face

        elif current_effect == "sift_keypoints":
            # Sift 關鍵點偵測
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            # 繪製關鍵點 (第三個參數 None 表示不使用現有影像作為輸出，讓它建立新的)
            processed_frame = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # 創建小畫面：縮放原始影片幀 (不再使用 processed_frame)
        pip_frame = cv2.resize(frame, (pip_width, pip_height), interpolation=cv2.INTER_AREA)
        
        # 在主畫面上繪製白色邊框
        cv2.rectangle(processed_frame, 
                      (pip_x - border_thickness, pip_y - border_thickness), 
                      (pip_x + pip_width + border_thickness, pip_y + pip_height + border_thickness), 
                      border_color, border_thickness)
        
        # 將小畫面嵌入到主畫面的指定位置
        processed_frame[pip_y:pip_y + pip_height, pip_x:pip_x + pip_width] = pip_frame
        
        # 創建動態幀數顯示文字
        frame_info_text = f"{frame_count + 1}/{total_frames} frames, FPS={fps}"

        # 在左上角添加特效名稱文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)  # 白色文字
        outline_color = (0, 0, 0)     # 黑色輪廓
        
        # 計算特效名稱文字大小
        (text_width1, text_height1), baseline1 = cv2.getTextSize(current_display_name, font, font_scale, font_thickness)
        # 計算幀數資訊文字大小
        (text_width2, text_height2), baseline2 = cv2.getTextSize(frame_info_text, font, font_scale * 0.7, font_thickness - 1)
        
        # 計算背景矩形的總體大小（取兩行文字的最大寬度）
        max_text_width = max(text_width1, text_width2)
        total_text_height = text_height1 + text_height2 + 10  # 10是行間距
        
        # 繪製半透明背景矩形（包含兩行文字）
        overlay = processed_frame.copy()
        cv2.rectangle(overlay, (10, 10), (20 + max_text_width, 30 + total_text_height + max(baseline1, baseline2)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
        
        # 繪製特效名稱文字輪廓（黑色）
        cv2.putText(processed_frame, current_display_name, (15, 15 + text_height1), 
                        font, font_scale, outline_color, font_thickness + 1, cv2.LINE_AA)
        
        # 繪製特效名稱文字（白色）
        cv2.putText(processed_frame, current_display_name, (15, 15 + text_height1), 
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # 計算第二行文字的Y座標位置
        second_line_y = 15 + text_height1 + text_height2 + 10  # 10是行間距
        
        # 繪製幀數資訊文字輪廓（黑色）
        cv2.putText(processed_frame, frame_info_text, (15, second_line_y), 
                        font, font_scale * 0.7, outline_color, font_thickness, cv2.LINE_AA)
        
        # 繪製幀數資訊文字（白色）
        cv2.putText(processed_frame, frame_info_text, (15, second_line_y), 
                        font, font_scale * 0.7, text_color, font_thickness - 1, cv2.LINE_AA)

        # 寫入處理後的幀
        out.write(processed_frame)

        # 顯示處理進度 (可選)
        if frame_count % fps == 0:
            elapsed_time = time.time() - start_time
            print(f"處理幀數：{frame_count}/{total_frames} (約 {elapsed_time:.2f} 秒)", end='\r')

        frame_count += 1

    print("\n影片處理完成！")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = "ぐでたまテーマソングMV(English subtitled).mp4"
    output_video = "gudetama_effects_output.mp4"
    apply_effects_to_video(input_video, output_video)
