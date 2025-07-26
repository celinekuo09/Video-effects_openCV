from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips


# 路徑設定
original_video = "ぐでたまテーマソングMV(English subtitled).mp4"        # 有聲音
processed_video = "gudetama_effects_output.mp4"                        # OpenCV特效、無聲

# 載入影片（新特效影片，無聲）
video_clip = VideoFileClip(processed_video)

# 載入原始影片，提取音訊
audio_clip = VideoFileClip(original_video).audio

# 將音訊設置到特效影片上
final_clip = video_clip.with_audio(audio_clip)

# 輸出最終結果，建議參數 preset="ultrafast"（快）、audio_codec="aac"
final_clip.write_videofile("gudetama_effects_with_audio.mp4",
                           codec="libx264",
                           audio_codec="aac",
                           preset="ultrafast")
