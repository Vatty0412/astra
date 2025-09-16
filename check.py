import cv2
import numpy as np
import imagehash
from PIL import Image
from collections import deque

class VideoTamperingDetector:
    """
    Consolidated, independent class to detect various forms of video tampering,
    including cuts, splices, loops, and unnatural environmental changes.
    """

    def detect_cuts_and_speed_changes(self, video_path, threshold=0.15, frame_skip=1):
        """
        Analyzes frame-to-frame differences (Mean Squared Error) to detect cuts.
        A sudden spike in the difference between consecutive frames suggests a cut.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'anomalies_found': True, 'flags': ['Video read error']}

        flags = []
        prev_frame = None
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_skip != 0:  # Skip frames for efficiency
                frame_number += 1
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (100, 100))  # Downscale for performance

            if prev_frame is not None:
                mse = np.mean((gray_frame.astype("float") - prev_frame.astype("float")) ** 2)
                diff_score = mse / (255.0 ** 2)  # ✅ normalized properly
                if diff_score > threshold:
                    flags.append(f"Potential cut or sudden change detected via MSE at frame {frame_number}")

            prev_frame = gray_frame
            frame_number += 1

        cap.release()
        return {'anomalies_found': len(flags) > 0, 'flags': flags}

    def analyze_optical_flow(self, video_path, outlier_threshold=2.5, frame_skip=1):
        """
        Analyzes optical flow to detect abrupt scene changes or splices.
        A sudden spike in the mean flow magnitude across frames is a strong indicator of a cut.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'anomalies_found': True, 'flags': ['Video read error']}

        flags = []
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return {'anomalies_found': False, 'flags': ['Video too short for optical flow analysis.']}
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flow_magnitudes = []
        frame_number = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_skip != 0:  # ✅ skip frames
                frame_number += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_magnitude = np.mean(magnitude)
            flow_magnitudes.append(mean_magnitude)
            
            prev_gray = gray
            frame_number += 1
        
        cap.release()

        if len(flow_magnitudes) > 10:
            mean_flow = np.mean(flow_magnitudes)
            std_dev_flow = np.std(flow_magnitudes)
            
            if std_dev_flow > 0:
                for i, mag in enumerate(flow_magnitudes):
                    z_score = (mag - mean_flow) / std_dev_flow
                    if z_score > outlier_threshold:
                        flags.append(f"Potential splice or cut detected via optical flow at frame {i+1}")
        
        return {'anomalies_found': len(flags) > 0, 'flags': flags}

    def detect_loops(self, video_path, sequence_length=30, hash_size=8, sampling_rate=1):
        """
        Detects loops by hashing video frames and looking for repeated sequences.
        Stores past sequences in a set, so loops are detected even if far apart.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'loops_found': True, 'flags': ['Video read error']}
        
        window = deque(maxlen=sequence_length)
        seen_sequences = set()
        flags = []
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % sampling_rate != 0:  # ✅ sample frames
                frame_number += 1
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_hash = str(imagehash.phash(Image.fromarray(gray), hash_size=hash_size))
            window.append(img_hash)

            if len(window) == sequence_length:
                seq = " ".join(window)
                if seq in seen_sequences:
                    flags.append(f"Potential looped video content detected around frame {frame_number}")
                    break  # stop at first detection (optional)
                seen_sequences.add(seq)

            frame_number += 1

        cap.release()
        return {'loops_found': len(flags) > 0, 'flags': flags}

    def detect_background_change(self, video_path, segments=3, threshold=0.7):
        """
        Compares histograms of frames from different segments of the video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'change_detected': True, 'flags': ['Video read error']}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < segments * 2:
            cap.release()
            return {'change_detected': False, 'flags': ['Video too short to analyze for splices.']}

        segment_histograms = []
        sample_frames = np.linspace(0, total_frames-1, segments).astype(int)  # ✅ smarter sampling
        
        for frame_index in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                background_sample = frame[h//4:3*h//4, w//4:3*w//4]
                hist = cv2.calcHist([background_sample], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                segment_histograms.append(hist)

        flags = []
        if len(segment_histograms) > 1:
            for i in range(len(segment_histograms) - 1):
                correlation = cv2.compareHist(segment_histograms[i], segment_histograms[i+1], cv2.HISTCMP_CORREL)
                if correlation < threshold:
                    flags.append(f"Major environmental change detected between segment {i+1} and {i+2}. Possible video splice.")
        
        cap.release()
        return {'change_detected': len(flags) > 0, 'flags': flags}

    def run_all_tampering_checks(self, video_path):
        """
        Runs all available video tampering checks and returns a consolidated report.
        """
        all_flags = []

        print("Starting tampering checks...")
        cut_results = self.detect_cuts_and_speed_changes(video_path, frame_skip=3)
        print(f"After cuts check: {cut_results}")
        if cut_results['anomalies_found']:
            all_flags.extend(cut_results['flags'])

        # Uncomment to enable optical flow check
        # flow_results = self.analyze_optical_flow(video_path, frame_skip=3)
        # print(f"After optical flow check: {flow_results}")
        # if flow_results['anomalies_found']:
        #     all_flags.extend(flow_results['flags'])

        loop_results = self.detect_loops(video_path, sequence_length=40, sampling_rate=3)
        print(f"After loop check: {loop_results}")
        if loop_results['loops_found']:
            all_flags.extend(loop_results['flags'])

        env_results = self.detect_background_change(video_path)
        print(f"After background change check: {env_results}")
        if env_results['change_detected']:
            all_flags.extend(env_results['flags'])

        print("All checks complete.")
        return {
            'tampering_detected': len(all_flags) > 0,
            'report': list(set(all_flags))
        }
        
if __name__ == '__main__':
    # This block allows the script to be run directly for testing.
    
    #Add video path.
    dummy_video_file = r"C:\Users\ARCHIT GOYAL\Downloads\6548176-hd_1920_1080_24fps.mp4"

    print("--- Testing Independent Video Tampering Detection ---")
    tampering_detector = VideoTamperingDetector()
    
    # Run all checks at once
    tampering_report = tampering_detector.run_all_tampering_checks(dummy_video_file)
    print(f"Consolidated Tampering Report: {tampering_report}")
