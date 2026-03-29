import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import csv
import time
import matplotlib.pyplot as plt


def rename_and_resize():
    with open ('new_dataset/Imagewise_Data.csv', newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        rows = dict(
            (row["Image Name"].strip(), row["Category"].strip())
            for row in reader
        )

        #print(rows)

        #time.sleep(10)

        healthy_index = 0
        benigh_index = 0
        opmd_index = 0
        oca_index = 0
        original_images_path = 'new_dataset/Images/'
        original_masks_path = 'rgb_masks/'
        
        dataset_sri_lanka_images_path = 'new_dataset_sri_lanka/images'
        dataset_sri_lanka_masks_path = 'new_dataset_sri_lanka/masks'

        with open("new_dataset/Imagewise_Data.csv", newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            rows = dict(
                (row["Image Name"].strip(), row["Category"].strip())
                for row in reader
            )

            for img_name in os.listdir(original_images_path):
                print(img_name[:-4], end=" ")

                class_found = rows[img_name[:-4]]
                print(class_found, end='\n')

                original_img = Image.open(original_images_path + img_name)
                original_mask = Image.open(original_masks_path + img_name[:-4] + '.png')
                '''
                plt.imshow(original_img)
                plt.axis("off")  # Hide axes
                plt.show()

                plt.imshow(original_mask)
                plt.axis("off")  # Hide axes
                plt.show()
                '''
                resized_img = original_img.resize((512, 512))
                resized_mask = original_mask.resize((512, 512))

                if class_found == "Healthy":
                    healthy_index += 1
                    index = healthy_index
                    print(f'healthy index: {healthy_index}')
                elif class_found == "Benign":
                    benigh_index += 1
                    index = benigh_index
                    print(f'benign index: {benigh_index}')
                elif class_found == "OPMD":
                    opmd_index += 1
                    index = opmd_index
                    print(f'opmd index: {opmd_index}')
                elif class_found == "OCA":
                    oca_index += 1
                    index = oca_index
                    print(f'oca index: {oca_index}')

                resized_img.save(dataset_sri_lanka_images_path + '/' + class_found + "_" + str(index) + ".png")
                resized_mask.save(dataset_sri_lanka_masks_path + '/' + class_found + "_" + str(index) + ".png")








                '''
                for row in rows:
                    #print(row["Image Name"], end='\n')
                    if row["Image Name"] == img_name[:-4]:
                        class_found = row['Category']
                        print(f'eh da classe {class_found}')
                        time.sleep(5)
                        break
                '''


def create_masks():
    # Paths
    coco_annotation_file = 'new_dataset/Annotation.json'
    imgs_path = 'new_dataset/Images/'
    output_dir = "rgb_masks"
    os.makedirs(output_dir, exist_ok=True)

    # Class colors
    CLASS_COLOR_MAP = {
        'Benign': (255, 255, 0),
        'OPMD': (0, 255, 0),
        'OCA': (255, 0, 0),
    }

    coco = COCO(coco_annotation_file)


    with open("new_dataset/Imagewise_Data.csv", newline="") as file:
        reader = csv.DictReader(file)

        rows = dict(
            (row["Image Name"].strip(), row["Category"].strip())
            for row in reader
        )

        #print(rows)
        #time.sleep(10)

        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            img_id = img_info['id']

            print(f'{file_name=}')
            print(f'{img_id=}')

            # Find current iteration class
            segmentation_class = rows[file_name[:-4]]
            '''
            segmentation_class = ""
            for row in rows:
                #print(row['Image Name'])
                #print(file_name)
                if row["Image Name"] != file_name[:-4]:
                    continue
                
                else:
                    segmentation_class += row['Category']
                    print(segmentation_class)
                    print(row['Category'])
                    break
            '''

            img = np.array(Image.open(imgs_path + file_name))

            print(img.shape) # (H, W, C)

            height, width, channels = img.shape

            #print(height)
            #print(width)

            # RGB mask
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            mask[:, :, 2] = 255

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                category_id = ann["category_id"]

                if category_id == 2:
                    continue


                if isinstance(ann["segmentation"], list) and isinstance(ann["segmentation"][0], (int, float)):
                    ann["segmentation"] = [ann["segmentation"]]


                if segmentation_class == 'Healthy':
                    continue
                else:
                    color = CLASS_COLOR_MAP[segmentation_class]

                # Convert COCO annotation to binary mask
                binary_mask = coco.annToMask(ann)

                # Paint pixels belonging to this object
                mask[binary_mask == 1] = color

            # Convert NumPy array → PIL Image
            mask_img = Image.fromarray(mask, mode="RGB")

            # Save (PNG recommended for masks)
            output_filename = os.path.splitext(img_info["file_name"])[0] + ".png"
            output_path = os.path.join(output_dir, output_filename)
            mask_img.save(output_path, format="PNG")


    print("RGB masks generated successfully!")


def main():
    rename_and_resize()


if __name__ == "__main__":
    main()