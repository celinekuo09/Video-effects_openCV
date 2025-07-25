import cv2
import numpy as np
import time

def apply_effects_to_video(input_video_path, output_video_path):
    """
    對影片應用一系列OpenCV特效。

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

    # 定義輸出影片的編碼器和檔案名
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或 'XVID' for .avi
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 定義每個特效的時間範圍 (秒)
    effects_timeline = {
        "original": (0, 8),
        "rotation_scale": (9, 18),
        "edge_detection": (19, 27),
        "morphological_gradient": (28, 36),
        "flip": (37, 46),
        "colormap": (47, 55),
        "dog_filter": (56, 64),
        "gamma_correction": (65, 74),
        "face_detection_mosaic": (75, 83),
        "sift_keypoints": (84, 92) # 修正為92秒結束，因為影片總長度可能不同
    }

    # 調整時間範圍為幀數
    effects_frame_ranges = {}
    for effect_name, (start_sec, end_sec) in effects_timeline.items():
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        effects_frame_ranges[effect_name] = (start_frame, end_frame)

    # 臉部偵測的 Haar Cascade 分類器
    # 請替換為你實際的 haarcascade_frontalface_default.xml 路徑
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # SIFT 偵測器
    sift = cv2.SIFT_create()

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_effect = "original"
        for effect_name, (start_frame, end_frame) in effects_frame_ranges.items():
            if start_frame <= frame_count <= end_frame:
                current_effect = effect_name
                break

        processed_frame = frame.copy()

        # 應用特效
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

            # 你可以選擇只顯示其中一個，或將它們疊加
            # # Sobel
            # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            # sobel = np.uint8(np.absolute(sobelx) + np.absolute(sobely))
            # processed_frame = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

            # # Laplacian
            # laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            # laplacian = np.uint8(np.absolute(laplacian))
            # processed_frame = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

        elif current_effect == "morphological_gradient":
            # 形態學梯度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed_frame = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        elif current_effect == "flip":
            # 翻轉 (水平和垂直翻轉)
            processed_frame = cv2.flip(frame, -1) # -1 表示水平和垂直翻轉

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