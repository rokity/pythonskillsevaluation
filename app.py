import sys
import cv2
import os

SIZE=(800,450)


def read_images_resize_save_output_dir(path, size,output_dir):
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image = cv2.imread(os.path.join(path, filename))
            if image.shape[0] > size[0] and image.shape[1] > size[1]:
                image = cv2.resize(image, size)
            cv2.imwrite(os.path.join(output_dir, filename), image)
    return images
    
    


def main():
    if len(sys.argv) != 7:
        print("Usage: python app.py --imagedir <path_to_images> --xmdir <path_to_xml> --outputdir <path_to_output>")
        sys.exit(1)
    image_dir = sys.argv[2]
    xml_dir = sys.argv[4]
    output_dir = sys.argv[6]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    read_images_resize_save_output_dir(image_dir, SIZE,output_dir)


if __name__ == '__main__':
    main()