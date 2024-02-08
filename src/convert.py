import glob
import os
import shutil
from urllib.parse import unquote, urlparse

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import file_exists, get_file_name, get_file_name_with_ext
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    dataset_path = "/home/alex/DATASETS/TODO/MuCeD"
    bbox_ext = ".txt"
    group_tag_name = "im id"
    batch_size = 30

    def create_ann(image_path):
        labels = []
        tags = []

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = 640  # image_np.shape[0]
        img_wight = 640  # image_np.shape[1]

        fold_value = image_path.split("/")[-4]
        fold_meta = fold_to_meta.get(fold_value)
        fold = sly.Tag(fold_meta)
        tags.append(fold)

        image_name = get_file_name(image_path)
        if image_name[:5] == "Image":
            im_id_value = fold_value + "_" + image_name.split("_")[1]
        else:
            im_id_value = fold_value + "_" + image_name.split("_")[0]

        group_id = sly.Tag(tag_id, value=im_id_value)
        tags.append(group_id)

        bbox_path = image_path.replace("images", "labels").replace(".jpg", ".txt")

        if file_exists(bbox_path):
            with open(bbox_path) as f:
                content = f.read().split("\n")

                for curr_data in content:
                    if len(curr_data) != 0:
                        curr_data = list(map(float, curr_data.split(" ")))
                        obj_class = idx_to_class[int(curr_data[0])]

                        left = int((curr_data[1] - curr_data[3] / 2) * img_wight)
                        right = int((curr_data[1] + curr_data[3] / 2) * img_wight)
                        top = int((curr_data[2] - curr_data[4] / 2) * img_height)
                        bottom = int((curr_data[2] + curr_data[4] / 2) * img_height)
                        rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                        label = sly.Label(rectangle, obj_class)
                        labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    obj_class_epithelial = sly.ObjClass("epithelial nuclei", sly.Rectangle)
    obj_class_intraepithelial = sly.ObjClass("intraepithelial lymphocyte", sly.Rectangle)

    fold0_meta = sly.TagMeta("fold 0", sly.TagValueType.NONE)
    fold1_meta = sly.TagMeta("fold 1", sly.TagValueType.NONE)
    fold2_meta = sly.TagMeta("fold 2", sly.TagValueType.NONE)
    fold3_meta = sly.TagMeta("fold 3", sly.TagValueType.NONE)
    fold4_meta = sly.TagMeta("fold 4", sly.TagValueType.NONE)

    fold_to_meta = {
        "0": fold0_meta,
        "1": fold1_meta,
        "2": fold2_meta,
        "3": fold3_meta,
        "4": fold4_meta,
    }

    tag_id = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)

    idx_to_class = {0: obj_class_epithelial, 1: obj_class_intraepithelial}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=list(idx_to_class.values()),
        tag_metas=[tag_id, fold0_meta, fold1_meta, fold2_meta, fold3_meta, fold4_meta],
    )
    api.project.update_meta(project.id, meta.to_json())
    api.project.images_grouping(id=project.id, enable=True, tag_name=group_tag_name)

    train_images = glob.glob(dataset_path + "/*/*/train/*.jpg")
    val_images = glob.glob(dataset_path + "/*/*/val/*.jpg")
    test_images = glob.glob(dataset_path + "/*/*/test/*.jpg")

    ds_name_to_pathes = {"train": train_images, "val": val_images, "test": test_images}

    for ds_name, images_pathes in ds_name_to_pathes.items():

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

        for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
            img_names_batch = [
                im_path.split("/")[-4] + "_" + get_file_name_with_ext(im_path)
                for im_path in img_pathes_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(img_names_batch))

    return project
