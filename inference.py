import os
import sys
import cv2
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms.functional_tensor')
video_folder = '/home1/nsbb/travail/DynamiCrafter/results/'
frame_folder = './results/frames/'
sr_frame_path = frame_folder+'sr/'
mp4_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
print(mp4_files)
for i in mp4_files:
    start_time = time.time()
    path = video_folder+i
    print(path)
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(frame_folder+'frame%d.jpg'%count,image)
        success,image = vidcap.read()
        print(frame_folder+'frame%d.jpg success!'%count)
        count += 1
        ref_time_st = time.time()
        os.system('python3 inference_realesrgan.py -i {}/frame{}.jpg -o {} -g 3'.format(frame_folder,count,sr_frame_path))
        ref_time_end = time.time()
        ref_time = ref_time_end - ref_time_st
        print(f'{ref_time:.2f} sec elapsed')
    print(count,count-2,count-1)
    frames = [f for f in os.listdir(sr_frame_path) if f.endswith('.jpg')]
    frames.sort()
    frame_path = os.path.join(sr_frame_path,frames[0])
    frame = cv2.imread(frame_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = i
    video = cv2.VideoWriter(output_video, fourcc, 12, (width, height))
    print(frames)

    for frame_file in frames:
        frame_path = os.path.join(sr_frame_path, frame_file)
        frame = cv2.imread(frame_path)
        print(frame_path,frame)

        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

    print(f'{output_video} created done!')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Execution time : {elapsed_time:.2f} seconds')
    break
