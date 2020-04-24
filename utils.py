from collections import defaultdict

def reorganize_annotations_by_filename(file):
    filename_by_image_id = {}
    for i in file['images']:
        filename_by_image_id[i['id']] = i['file_name']

    annotations_by_filename = defaultdict(list)
    for a in file['annotations']:
        filename = filename_by_image_id[a['image_id']]
        annotations_by_filename[filename].append(a)

    return annotations_by_filename 