import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 打不开摄像头，检查设备！")
        return

    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("✅ 启动，按 q 退出；会打印全部 33 个关键点信息")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            # 绘制骨架与 33 个关键点
            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw_style.get_default_pose_landmarks_style()
            )

            points = []
            low_conf_count = 0
            for idx, lm in enumerate(res.pose_landmarks.landmark):
                points.append({
                    "idx": idx,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })
                if lm.visibility < 0.35:
                    low_conf_count += 1

            total = len(points)
            bad_rate = low_conf_count / total if total else 0
            print(
                f"\n【检测】一共 {total} 点｜低置信数量：{low_conf_count}｜异常占比：{bad_rate * 100:.1f}%"
            )

        cv2.imshow("摄像头 - 33 人体关键点（单个人）", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("已退出")


if __name__ == "__main__":
    main()