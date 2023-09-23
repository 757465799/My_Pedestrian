import shutil
import cv2 as cv

sets = ['train', 'val', 'test']
for image_set in sets:
    image_ids = open('../LabelmeToCoco/%s.txt' % (image_set)).read().strip().split()
    for image_id in image_ids:
        img = cv.imread('../Dataset/labelme_image/%s.jpg' % (image_id))
        json = '../Dataset/labelme_json/%s.json' % (image_id)
        cv.imwrite('../LabelmeToCoco/images/%s/%s.jpg' % (image_set, image_id), img)
        cv.imwrite('../LabelmeToCoco/labels/%s/%s.jpg' % (image_set, image_id), img)
        shutil.copy(json, '../LabelmeToCoco/labels/%s/%s.json' % (image_set, image_id))
print("Finish!")

