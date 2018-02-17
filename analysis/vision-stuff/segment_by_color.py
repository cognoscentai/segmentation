# runs vision-reference-code/segment.cpp to segment images by color

import glob
import os

CURR_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
INPUT_IMG_DIR = CURR_DIR + 'COCO-ppm/'
OUTPUT_IMG_DIR = CURR_DIR + 'color-segmented-images/'
CODE_DIR = CURR_DIR + 'vision-reference-code/'


def compile_code():
    os.system('cd {}; make'.format(CODE_DIR))


def segment_image_by_color(
    sigma, k, m,
    input_img_path, output_img_path
):
    os.system(
        'cd {}; ./segment {} {} {} {} {}'.format(
            CODE_DIR,
            sigma, k, m,
            input_img_path, output_img_path
        )
    )


def segment_all(k=500):
    compile_code()
    for input_img_path in glob.glob('{}*'.format(INPUT_IMG_DIR)):
        img_name = input_img_path.split('/')[-1].split('.')[0]
        outdir = OUTPUT_IMG_DIR + str(k)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        output_img_path = '{}/{}.png'.format(outdir, img_name)
        # print output_img_path
        if img_name:
            # currently using same params for all images
            sigma = 0.5
            m = 20
        segment_image_by_color(sigma, k, m, input_img_path, output_img_path)


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)
    for k in range(100,550,50):
        segment_all(k)
