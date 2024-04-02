import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FFMPEG.')
    parser.add_argument('-v', '--video_file', type=str, help='The path to the video file')

    # sys.argv.extend(["-v", "convertedExterndisk0_Ch2_20240305170000_20240305180000Converted.mp4"])
    sys.argv.extend(["-v", "vid.webm"])

    args = parser.parse_args()


path = "results/vid.webm/vid.webm_%010d.jpg"
output_path = "ffmpeg_output/vid_webm.mp4"

fps = 30
subprocess.Popen(f'ffmpeg -framerate {fps} -i {path} -c:v libx264 -r {fps} -pix_fmt yuv420p output.mp4', shell=True)
