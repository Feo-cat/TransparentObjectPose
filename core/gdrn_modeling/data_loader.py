# -*- coding: utf-8 -*-
import copy
import hashlib
import logging
import os
import math
import os.path as osp
import pickle
import random

import cv2
import mmcv
import numpy as np
import ref
import torch
import torch.multiprocessing as mp
from core.base_data_loader import Base_DatasetFromList
from core.utils.augment import AugmentRGB
from core.utils.data_utils import (
    crop_resize_by_warp_affine,
    farthest_point_sampling_np,
    get_2d_coord_np,
    read_image_cv2,
    xyz_to_region,
)
from core.utils.dataset_utils import (
    filter_empty_dets,
    filter_invalid_in_dataset_dicts,
    flat_dataset_dicts,
    flat_dataset_dicts_list,
    load_detections_into_dataset,
    my_build_batch_data_loader,
    trivial_batch_collator,
)
from core.utils.my_distributed_sampler import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from core.utils.rot_reps import mat_to_ortho6d_np
from core.utils.ssd_color_transform import ColorAugSSDTransform
from core.utils.utils import egocentric_to_allocentric
from core.utils import quaternion_lf, lie_algebra
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances, Keypoints, PolygonMasks
from detectron2.utils.logger import log_first_n
from fvcore.common.file_io import PathManager
from lib.pysixd import inout, misc
from lib.utils.mask_utils import cocosegm2mask, get_edge
from lib.utils.utils import dprint, lazy_property
from transforms3d.quaternions import mat2quat

from .dataset_factory import register_datasets

logger = logging.getLogger(__name__)


def infer_depth_npy_from_rgb_file(rgb_file):
    """Infer labsim-style depth npy path from rgb png path.

    Example:
    .../rgb/000000/000000.png -> .../depth/000000/000000.npy
    """
    depth_file = rgb_file.replace("/rgb/", "/depth/")
    return osp.splitext(depth_file)[0] + ".npy"


def transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    """
    NOTE: Adapted from detection_utils.
    Apply transforms to box, segmentation, keypoints, etc. of annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    im_H, im_W = image_size
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = np.array(transforms.apply_box([bbox])[0])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # NOTE: here we transform segms to binary masks (interp is nearest by default)
        # vis_mask = (cocosegm2mask(annotation["segmentation"], h=im_H, w=im_W)*255).astype(np.uint8)
        # from PIL import Image
        # Image.fromarray(vis_mask).save(f"/home/renchengwei/GDR-Net/debug/vis_mask_orig.png")
        
        mask = transforms.apply_segmentation(cocosegm2mask(annotation["segmentation"], h=im_H, w=im_W))
        annotation["segmentation"] = mask

    if "keypoints" in annotation:
        keypoints = utils.transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    if "centroid_2d" in annotation:
        annotation["centroid_2d"] = transforms.apply_coords(np.array(annotation["centroid_2d"]).reshape(1, 2)).flatten()

    return annotation


def build_gdrn_augmentation(cfg, is_train):
    """Create a list of :class:`Augmentation` from config. when training 6d
    pose, cannot flip.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        # augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


class GDRN_DatasetFromList(Base_DatasetFromList):
    """NOTE: we can also use the default DatasetFromList and
    implement a similar custom DataMapper,
    but it is harder to implement some features relying on other dataset dicts
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/common.py
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, cfg, split, lst: list, copy: bool = True, serialize: bool = True, flatten=True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self.augmentation = build_gdrn_augmentation(cfg, is_train=(split == "train"))
        if cfg.INPUT.COLOR_AUG_PROB > 0 and cfg.INPUT.COLOR_AUG_TYPE.lower() == "ssd":
            self.augmentation.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            logging.getLogger(__name__).info("Color augmnetation used in training: " + str(self.augmentation[-1]))
        # fmt: off
        self.img_format = cfg.INPUT.FORMAT  # default BGR
        self.with_depth = cfg.INPUT.WITH_DEPTH
        self.aug_depth = cfg.INPUT.AUG_DEPTH
        # NOTE: color augmentation config
        self.color_aug_prob = cfg.INPUT.COLOR_AUG_PROB
        self.color_aug_type = cfg.INPUT.COLOR_AUG_TYPE
        self.color_aug_code = cfg.INPUT.COLOR_AUG_CODE
        # dinov3 image mean and std
        # self.dinov3_image_mean = torch.tensor([0.485, 0.456, 0.406])
        # self.dinov3_image_std = torch.tensor([0.229, 0.224, 0.225])
        self.dinov3_image_mean = np.array([0.485, 0.456, 0.406])
        self.dinov3_image_std = np.array([0.229, 0.224, 0.225])
        # fmt: on
        self.cfg = cfg
        self.split = split  # train | val | test
        if split == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        # ------------------------
        # common model infos
        self.fps_points = {}
        self.model_points = {}
        self.extents = {}
        self.sym_infos = {}
        # ----------------------------------------------------
        self.flatten = flatten
        self._lst = flat_dataset_dicts_list(lst) if flatten else lst
        # ----------------------------------------------------
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger.info("Serializing {} elements to byte tensors and concatenating them all ...".format(len(self._lst)))
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def _get_fps_points(self, dataset_name, with_center=False):
        """convert to label based keys.

        # TODO: get models info similarly
        """
        if dataset_name in self.fps_points:
            return self.fps_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg
        num_fps_points = cfg.MODEL.CDPN.ROT_HEAD.NUM_REGIONS
        cur_fps_points = {}
        loaded_fps_points = data_ref.get_fps_points()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            if with_center:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"]
            else:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"][:-1]
        self.fps_points[dataset_name] = cur_fps_points
        return self.fps_points[dataset_name]

    def _get_model_points(self, dataset_name):
        """convert to label based keys."""
        if dataset_name in self.model_points:
            return self.model_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_model_points = {}
        num = np.inf
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            cur_model_points[i] = pts = model["pts"]
            if pts.shape[0] < num:
                num = pts.shape[0]

        num = min(num, cfg.MODEL.CDPN.PNP_NET.NUM_PM_POINTS)
        for i in range(len(cur_model_points)):
            cur_model_points[i] = farthest_point_sampling_np(cur_model_points[i], num_samples=num, init_center=True)

        self.model_points[dataset_name] = cur_model_points
        return self.model_points[dataset_name]

    def _get_extents(self, dataset_name):
        """label based keys."""
        if dataset_name in self.extents:
            return self.extents[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        try:
            ref_key = dset_meta.ref_key
        except:
            # FIXME: for some reason, in distributed training, this need to be re-registered
            register_datasets([dataset_name])
            dset_meta = MetadataCatalog.get(dataset_name)
            ref_key = dset_meta.ref_key

        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_extents = {}
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            # model = inout.load_ply(model_path)
            pts = model["pts"]
            xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
            ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
            zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            cur_extents[i] = np.array([size_x, size_y, size_z], dtype="float32")

        self.extents[dataset_name] = cur_extents
        return self.extents[dataset_name]

    def _get_sym_infos(self, dataset_name):
        """label based keys."""
        if dataset_name in self.sym_infos:
            return self.sym_infos[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_sym_infos = {}
        loaded_models_info = data_ref.get_models_info()
        if not hasattr(self, "model_infos"):
            self.model_infos = loaded_models_info
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_info = loaded_models_info[str(obj_id)]
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
                sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
            else:
                sym_info = None
            cur_sym_infos[i] = sym_info

        self.sym_infos[dataset_name] = cur_sym_infos
        return self.sym_infos[dataset_name]

    def read_data(self, dataset_dict):
        """load image and annos random shift & scale bbox; crop, rescale."""
        cfg = self.cfg
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        dataset_name = dataset_dict["dataset_name"]

        image = read_image_cv2(dataset_dict["file_name"], format=self.img_format)
        # should be consistent with the size in dataset_dict
        utils.check_image_size(dataset_dict, image)
        im_H_ori, im_W_ori = image.shape[:2]
        depth = None
        if self.with_depth:
            depth_file_raw = dataset_dict.get("depth_file", "")
            inferred_depth_file = infer_depth_npy_from_rgb_file(dataset_dict["file_name"])
            candidate_depth_files = []
            if depth_file_raw:
                candidate_depth_files.append(depth_file_raw)
                # Compat: many old dataset records point to .png depth, while labsim now stores .npy.
                candidate_depth_files.append(osp.splitext(depth_file_raw)[0] + ".npy")
            candidate_depth_files.append(inferred_depth_file)

            depth_file = ""
            for cand_path in candidate_depth_files:
                if cand_path and osp.exists(cand_path):
                    depth_file = cand_path
                    break

            if not depth_file:
                raise FileNotFoundError(
                    "Depth file does not exist for RGB {}. Tried: {}".format(
                        dataset_dict["file_name"],
                        candidate_depth_files,
                    )
                )
            depth = np.load(depth_file)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[:, :, 0]
            if depth.ndim != 2:
                raise ValueError(f"Expect depth to be HxW (or HxWx1), got shape={depth.shape}, file={depth_file}")
            if depth.shape[:2] != (im_H_ori, im_W_ori):
                raise ValueError(
                    f"Depth/RGB size mismatch for {dataset_dict['file_name']}: "
                    f"rgb={(im_H_ori, im_W_ori)}, depth={depth.shape[:2]}, depth_file={depth_file}"
                )
            dataset_dict["depth_file"] = depth_file

        # currently only replace bg for train ###############################
        if self.split == "train":
            # some synthetic data already has bg, img_type should be real or something else but not syn
            img_type = dataset_dict.get("img_type", "real")
            # import pdb; pdb.set_trace()
            if img_type == "syn":
                log_first_n(logging.WARNING, "replace bg", n=10)
                assert "segmentation" in dataset_dict["inst_infos"]
                mask = cocosegm2mask(dataset_dict["inst_infos"]["segmentation"], im_H_ori, im_W_ori)
                image, mask_trunc = self.replace_bg(image.copy(), mask, return_mask=True)
            else:  # real image
                if np.random.rand() < cfg.INPUT.CHANGE_BG_PROB:
                    log_first_n(logging.WARNING, "replace bg for real", n=10)
                    assert "segmentation" in dataset_dict["inst_infos"]
                    mask = cocosegm2mask(dataset_dict["inst_infos"]["segmentation"], im_H_ori, im_W_ori)
                    image, mask_trunc = self.replace_bg(image.copy(), mask, return_mask=True)
                else:
                    mask_trunc = None

        # NOTE: maybe add or change color augment here ===================================
        if self.split == "train" and self.color_aug_prob > 0 and self.color_augmentor is not None:
            if np.random.rand() < self.color_aug_prob:
                if cfg.INPUT.COLOR_AUG_SYN_ONLY and img_type not in ["real"]:
                    image = self._color_aug(image, self.color_aug_type)
                else:
                    image = self._color_aug(image, self.color_aug_type)

        # other transforms (mainly geometric ones);
        # for 6d pose task, flip is now allowed in general except for some 2d keypoints methods
        image, transforms = T.apply_augmentations(self.augmentation, image)
        if depth is not None:
            depth = transforms.apply_segmentation(depth).astype(np.float32)
            dataset_dict["input_depths"] = torch.as_tensor(depth, dtype=torch.float32).contiguous()
        # add input image for debug
        image_copy = image.copy()
        # convert to RGB
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_copy = image_copy.transpose(2, 0, 1) # HWC -> CHW
        image_copy = image_copy / 255.
        mean = self.dinov3_image_mean[:, None, None]
        std  = self.dinov3_image_std[:, None, None]
        image_copy = (image_copy - mean) / std
        image_copy = torch.as_tensor(image_copy, dtype=torch.float32).contiguous()
        dataset_dict["input_images"] = image_copy
        im_H, im_W = image_shape = image.shape[:2]  # h, w

        # NOTE: scale camera intrinsic if necessary ================================
        scale_x = im_W / im_W_ori
        scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
        if "cam" in dataset_dict:
            if im_W != im_W_ori or im_H != im_H_ori:
                dataset_dict["cam"][0] *= scale_x
                dataset_dict["cam"][1] *= scale_y
            K = dataset_dict["cam"].astype("float32")
            dataset_dict["cam"] = torch.as_tensor(K)

        input_res = cfg.MODEL.CDPN.BACKBONE.INPUT_RES
        out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

        # CHW -> HWC
        coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)

        #################################################################################
        # -----------------------------if not train-------------------------------------
        if self.split != "train":
            # Process each instance separately to match training data format
            test_bbox_type = cfg.TEST.TEST_BBOX_TYPE
            if test_bbox_type == "gt":
                bbox_key = "bbox"
            else:
                bbox_key = f"bbox_{test_bbox_type}"
            assert not self.flatten, "Do not use flattened dicts for test!"

            inst_dicts = []
            for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):
                inst_dict = {}

                # --- shared image-level fields (same format as training) ---
                inst_dict["dataset_name"] = dataset_dict["dataset_name"]
                inst_dict["scene_im_id"] = dataset_dict["scene_im_id"]
                inst_dict["file_name"] = dataset_dict["file_name"]
                inst_dict["height"] = im_H
                inst_dict["width"] = im_W
                inst_dict["cam"] = dataset_dict["cam"]  # tensor (3, 3)
                inst_dict["input_images"] = dataset_dict["input_images"]  # tensor (C, H, W)
                if "input_depths" in dataset_dict:
                    inst_dict["input_depths"] = dataset_dict["input_depths"]  # tensor (H, W)

                # --- instance-level fields (same format as training) ---
                inst_dict["inst_id"] = inst_i
                inst_dict["model_info"] = inst_infos["model_info"]

                if "segmentation" in inst_infos:
                    inst_anno_for_mask = transform_instance_annotations(
                        copy.deepcopy(inst_infos), transforms, image_shape, keypoint_hflip_indices=None
                    )
                    inst_dict["input_obj_masks"] = torch.as_tensor(
                        inst_anno_for_mask["segmentation"].astype("float32")
                    ).contiguous()

                if "pose" in inst_infos:
                    inst_dict["ego_rot"] = torch.as_tensor(inst_infos["pose"][:3, :3].astype("float32"))
                if "trans" in inst_infos:
                    inst_dict["trans"] = torch.as_tensor(inst_infos["trans"].astype("float32"))

                roi_cls = inst_infos["category_id"]
                inst_dict["roi_cls"] = roi_cls  # int, like training
                inst_dict["score"] = inst_infos.get("score", 1.0)  # float

                # extent
                roi_extent = self._get_extents(dataset_name)[roi_cls]
                inst_dict["roi_extent"] = torch.as_tensor(np.array(roi_extent), dtype=torch.float32)

                # bbox — align to patch grid (same as training aug_bbox, without random augmentation)
                bbox = BoxMode.convert(inst_infos[bbox_key], inst_infos["bbox_mode"], BoxMode.XYXY_ABS)
                bbox = np.array(transforms.apply_box([bbox])[0])
                x1, y1, x2, y2 = bbox
                bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
                bw = max(x2 - x1, 1)
                bh = max(y2 - y1, 1)
                scale = max(bh, bw) * cfg.INPUT.DZI_PAD_SCALE
                scale = min(scale, max(im_H, im_W)) * 1.0


                if cfg.INPUT.DZI_PATCH_GRID_ADSORPTION:
                    # re-shift bbox to align with patch grid
                    patch_size = 16
                    xmin = bbox_center[0] - scale / 2
                    ymin = bbox_center[1] - scale / 2
                    xmax = bbox_center[0] + scale / 2
                    ymax = bbox_center[1] + scale / 2
                    # convert to patch coordinates
                    px_min = np.floor(xmin / patch_size)
                    py_min = np.floor(ymin / patch_size)
                    px_max = np.ceil(xmax / patch_size)
                    py_max = np.ceil(ymax / patch_size)
                    # enforce square & even number of patches
                    pw = px_max - px_min
                    ph = py_max - py_min
                    p = max(pw, ph)
                    if p % 2 == 1:
                        p += 1
                    cx_p = (px_min + px_max) / 2
                    cy_p = (py_min + py_max) / 2
                    px_min = cx_p - p / 2
                    px_max = cx_p + p / 2
                    py_min = cy_p - p / 2
                    py_max = cy_p + p / 2
                    # back to pixel coordinates
                    xmin = px_min * patch_size
                    xmax = px_max * patch_size
                    ymin = py_min * patch_size
                    ymax = py_max * patch_size
                    # update center, scale, bbox, bw, bh after alignment
                    bbox_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2], dtype=np.float32)
                    scale = float(xmax - xmin)  # guaranteed multiple of patch_size
                    bw = max(xmax - xmin, 1)
                    bh = max(ymax - ymin, 1)

                inst_dict["bbox_center"] = torch.as_tensor(bbox_center, dtype=torch.float32)
                inst_dict["scale"] = scale  # float, like training
                inst_dict["bbox"] = bbox
                inst_dict["bbox_mode"] = BoxMode.XYXY_ABS
                inst_dict["roi_wh"] = torch.as_tensor(np.array([bw, bh], dtype=np.float32))
                inst_dict["resize_ratio"] = out_res / scale  # float scalar, like training

                # ROI image — (C, H, W), like training
                roi_img = crop_resize_by_warp_affine(
                    image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1)
                roi_img = self.normalize_image(cfg, roi_img)
                inst_dict["roi_img"] = torch.as_tensor(roi_img.astype("float32")).contiguous()

                # roi_coord_2d — (2, out_H, out_W), like training
                roi_coord_2d = crop_resize_by_warp_affine(
                    coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1)
                inst_dict["roi_coord_2d"] = torch.as_tensor(roi_coord_2d.astype("float32")).contiguous()

                # patch_mask — (H//16, W//16), like training
                patch_mask = bbox_to_patch_mask(bbox_center, scale, 16, im_H // 16, im_W // 16)
                inst_dict["patch_mask"] = torch.as_tensor(patch_mask.astype("bool")).contiguous()

                inst_dicts.append(inst_dict)

            return inst_dicts  # list of single-instance dicts, each matching training format
        #######################################################################################
        # -----------------------------if train-------------------------------------
        # NOTE: currently assume flattened dicts for train
        assert self.flatten, "Only support flattened dicts for train now"
        inst_infos = dataset_dict.pop("inst_infos")
        dataset_dict["roi_cls"] = roi_cls = inst_infos["category_id"]

        # extent
        roi_extent = self._get_extents(dataset_name)[roi_cls]
        dataset_dict["roi_extent"] = torch.as_tensor(np.array(roi_extent), dtype=torch.float32)

        # load xyz =======================================================
        xyz_info = mmcv.load(inst_infos["xyz_path"])
        x1, y1, x2, y2 = xyz_info["xyxy"]
        # float16 does not affect performance (classification/regresion)
        xyz_crop = xyz_info["xyz_crop"]
        xyz = np.zeros((im_H, im_W, 3), dtype=np.float32)
        xyz[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop
        # NOTE: full mask
        mask_obj = ((xyz[:, :, 0] != 0) | (xyz[:, :, 1] != 0) | (xyz[:, :, 2] != 0)).astype(np.bool).astype(np.float32)
        if cfg.INPUT.SMOOTH_XYZ:
            xyz = self.smooth_xyz(xyz)

        if cfg.TRAIN.VIS:
            xyz = self.smooth_xyz(xyz)

        # override bbox info using xyz_infos
        inst_infos["bbox"] = [x1, y1, x2, y2]
        inst_infos["bbox_mode"] = BoxMode.XYXY_ABS

        # USER: Implement additional transformations if you have other types of data
        # inst_infos.pop("segmentation")  # NOTE: use mask from xyz
        anno = transform_instance_annotations(inst_infos, transforms, image_shape, keypoint_hflip_indices=None)
        dataset_dict["input_obj_masks"] = torch.as_tensor(anno["segmentation"].astype("float32")).contiguous()

        # mask noise augmentation: perturb mask → recompute bbox (simulates noisy inference masks)
        noisy_mask = None
        if cfg.INPUT.MASK_NOISE_AUG_PROB > 0:
            if np.random.rand() < cfg.INPUT.MASK_NOISE_AUG_PROB:
                noisy_mask = self.augment_mask_noise(anno["segmentation"].astype("float32"), cfg)
                dataset_dict["noisy_obj_mask"] = torch.as_tensor(noisy_mask).contiguous()
            else:
                # Always store key when feature is enabled (needed for consistent batching)
                dataset_dict["noisy_obj_mask"] = torch.as_tensor(
                    anno["segmentation"].astype("float32")
                ).contiguous()

        # augment bbox ===================================================
        # NOTE: noisy_mask is intentionally NOT used to recompute bbox.
        # spur/blob noise shifts pixel extremes dramatically, producing ROI
        # drift far worse than any real inference artifact. In AR inference
        # the ROI is stabilised by EMA tracking from the previous chunk, so
        # the bbox is always stable even when the mask is noisy. We only use
        # noisy_mask for mask_attention inside the (stable) ROI crop.
        bbox_xyxy = anno["bbox"]
        bbox_center, scale = self.aug_bbox(cfg, bbox_xyxy, im_H, im_W, patch_size=16)
        bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
        bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)
        
        patch_mask = bbox_to_patch_mask(bbox_center, scale, 16, im_H // 16, im_W // 16)
        # # h and w should be the same
        # ys, xs = np.where(patch_mask)
        # patch_h, patch_w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        # import pdb; pdb.set_trace()
        
        # # visualize patch mask
        # from PIL import Image
        # Image.fromarray((patch_mask*255).astype(np.uint8)).save(f"/home/renchengwei/GDR-Net/debug/patch_mask.png")
        # vis_img = np.zeros((im_H, im_W, 3), dtype=np.uint8)
        # xmin, ymin, xmax, ymax = bbox_center[0] - scale / 2, bbox_center[1] - scale / 2, bbox_center[0] + scale / 2, bbox_center[1] + scale / 2
        # cv2.rectangle(vis_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        # Image.fromarray(vis_img.astype(np.uint8)).save(f"/home/renchengwei/GDR-Net/debug/vis_img.png")


        # CHW, float32 tensor
        ## roi_image ------------------------------------
        roi_img = crop_resize_by_warp_affine(
            image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        # # draw bbox
        # print("--------------------------------")
        # bbox_image = image.copy()
        # cv2.rectangle(bbox_image, (int(bbox_xyxy[0]), int(bbox_xyxy[1])), (int(bbox_xyxy[2]), int(bbox_xyxy[3])), (0, 0, 255), 2)
        # cv2.putText(bbox_image, f"Rack: {int(bbox_xyxy[2] - bbox_xyxy[0])}x{int(bbox_xyxy[3] - bbox_xyxy[1])}", (int(bbox_xyxy[0]), int(bbox_xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.imwrite(f"/home/renchengwei/GDR-Net/debug/roi_gt_img.png", bbox_image)
        
        # patch_image = image.copy()
        # cv2.rectangle(patch_image, (int(bbox_center[0] - scale / 2), int(bbox_center[1] - scale / 2)), (int(bbox_center[0] + scale / 2), int(bbox_center[1] + scale / 2)), (0, 0, 255), 2)
        # cv2.putText(patch_image, f"Rack: {int(scale)}", (int(bbox_center[0] - scale / 2), int(bbox_center[1] - scale / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.imwrite(f"/home/renchengwei/GDR-Net/debug/roi_img.png", patch_image)
        
        # cv2.imwrite(f"/home/renchengwei/GDR-Net/debug/roi_patch_img.png", roi_img.transpose(1, 2, 0))
        # import pdb; pdb.set_trace()


        roi_img = self.normalize_image(cfg, roi_img)

        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        ## roi_mask ---------------------------------------
        # (mask_trunc < mask_visib < mask_obj)
        mask_visib = anno["segmentation"].astype("float32") * mask_obj
        if mask_trunc is None:
            mask_trunc = mask_visib
        else:
            mask_trunc = mask_visib * mask_trunc.astype("float32")

        if cfg.TRAIN.VIS:
            mask_xyz_interp = cv2.INTER_LINEAR
        else:
            mask_xyz_interp = cv2.INTER_NEAREST

        # maybe truncated mask (true mask for rgb)
        roi_mask_trunc = crop_resize_by_warp_affine(
            mask_trunc[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        # use original visible mask to calculate xyz loss (try full obj mask?)
        roi_mask_visib = crop_resize_by_warp_affine(
            mask_visib[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        roi_mask_obj = crop_resize_by_warp_affine(
            mask_obj[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        # vis_mask = (anno["segmentation"]*255).astype(np.uint8)
        # from PIL import Image
        # Image.fromarray(vis_mask).save(f"/home/renchengwei/GDR-Net/debug/vis_mask.png")
        # import pdb; pdb.set_trace()
        
        # vis_mask = (mask_visib*255).astype(np.uint8)
        # from PIL import Image
        # Image.fromarray(vis_mask).save(f"/home/renchengwei/GDR-Net/debug/vis_mask_visib.png")
        # vis_mask = (mask_obj*255).astype(np.uint8)
        # Image.fromarray(vis_mask).save(f"/home/renchengwei/GDR-Net/debug/vis_mask_obj.png")
        # roi_mask_obj = (roi_mask_obj*255).astype(np.uint8)
        # Image.fromarray(roi_mask_obj).save(f"/home/renchengwei/GDR-Net/debug/roi_mask_obj.png")
        # import pdb; pdb.set_trace()

        ## roi_xyz ----------------------------------------------------
        roi_xyz = crop_resize_by_warp_affine(xyz, bbox_center, scale, out_res, interpolation=mask_xyz_interp)

        fps_points = None
        # region label
        if r_head_cfg.NUM_REGIONS > 1:
            fps_points = self._get_fps_points(dataset_name)[roi_cls]
            roi_region = xyz_to_region(roi_xyz, fps_points)  # HW
            dataset_dict["roi_region"] = torch.as_tensor(roi_region.astype(np.int32)).contiguous()

        roi_xyz = roi_xyz.transpose(2, 0, 1)  # HWC-->CHW
        # normalize xyz to [0, 1] using extent
        # print(roi_xyz.min(), roi_xyz.max())
        roi_xyz[0] = roi_xyz[0] / roi_extent[0] + 0.5
        roi_xyz[1] = roi_xyz[1] / roi_extent[1] + 0.5
        roi_xyz[2] = roi_xyz[2] / roi_extent[2] + 0.5
        # print(roi_xyz.min(), roi_xyz.max(), roi_extent)
        # print(np.percentile(roi_xyz[0], 2), np.percentile(roi_xyz[0], 98))
        # import pdb; pdb.set_trace()

        if ("CE" in r_head_cfg.XYZ_LOSS_TYPE) or ("cls" in cfg.MODEL.CDPN.NAME):  # convert target to int for cls
            # assume roi_xyz has been normalized in [0, 1]
            roi_xyz_bin = np.zeros_like(roi_xyz)
            roi_x_norm = roi_xyz[0]
            roi_x_norm[roi_x_norm < 0] = 0  # clip
            roi_x_norm[roi_x_norm > 0.999999] = 0.999999
            # [0, BIN-1]
            roi_xyz_bin[0] = np.asarray(roi_x_norm * r_head_cfg.XYZ_BIN, dtype=np.uint8)

            roi_y_norm = roi_xyz[1]
            roi_y_norm[roi_y_norm < 0] = 0
            roi_y_norm[roi_y_norm > 0.999999] = 0.999999
            roi_xyz_bin[1] = np.asarray(roi_y_norm * r_head_cfg.XYZ_BIN, dtype=np.uint8)

            roi_z_norm = roi_xyz[2]
            roi_z_norm[roi_z_norm < 0] = 0
            roi_z_norm[roi_z_norm > 0.999999] = 0.999999
            roi_xyz_bin[2] = np.asarray(roi_z_norm * r_head_cfg.XYZ_BIN, dtype=np.uint8)

            # the last bin is for bg
            roi_masks = {"trunc": roi_mask_trunc, "visib": roi_mask_visib, "obj": roi_mask_obj}
            roi_mask_xyz = roi_masks[r_head_cfg.XYZ_LOSS_MASK_GT]
            roi_xyz_bin[0][roi_mask_xyz == 0] = r_head_cfg.XYZ_BIN
            roi_xyz_bin[1][roi_mask_xyz == 0] = r_head_cfg.XYZ_BIN
            roi_xyz_bin[2][roi_mask_xyz == 0] = r_head_cfg.XYZ_BIN

            if "CE" in r_head_cfg.XYZ_LOSS_TYPE:
                dataset_dict["roi_xyz_bin"] = torch.as_tensor(roi_xyz_bin.astype("uint8")).contiguous()
            if "/" in r_head_cfg.XYZ_LOSS_TYPE and len(r_head_cfg.XYZ_LOSS_TYPE.split("/")[1]) > 0:
                dataset_dict["roi_xyz"] = torch.as_tensor(roi_xyz.astype("float32")).contiguous()
        else:
            dataset_dict["roi_xyz"] = torch.as_tensor(roi_xyz.astype("float32")).contiguous()

        # pose targets ----------------------------------------------------------------------
        pose = inst_infos["pose"]
        allo_pose = egocentric_to_allocentric(pose)
        quat = inst_infos["quat"]
        allo_quat = mat2quat(allo_pose[:3, :3])

        # ====== actually not needed ==========
        if pnp_net_cfg.ROT_TYPE == "allo_quat":
            dataset_dict["allo_quat"] = torch.as_tensor(allo_quat.astype("float32"))
        elif pnp_net_cfg.ROT_TYPE == "ego_quat":
            dataset_dict["ego_quat"] = torch.as_tensor(quat.astype("float32"))
        # rot6d
        elif pnp_net_cfg.ROT_TYPE == "ego_rot6d":
            dataset_dict["ego_rot6d"] = torch.as_tensor(mat_to_ortho6d_np(pose[:3, :3].astype("float32")))
        elif pnp_net_cfg.ROT_TYPE == "allo_rot6d":
            dataset_dict["allo_rot6d"] = torch.as_tensor(mat_to_ortho6d_np(allo_pose[:3, :3].astype("float32")))
        # log quat
        elif pnp_net_cfg.ROT_TYPE == "ego_log_quat":
            dataset_dict["ego_log_quat"] = quaternion_lf.qlog(torch.as_tensor(quat.astype("float32"))[None])[0]
        elif pnp_net_cfg.ROT_TYPE == "allo_log_quat":
            dataset_dict["allo_log_quat"] = quaternion_lf.qlog(torch.as_tensor(allo_quat.astype("float32"))[None])[0]
        # lie vec
        elif pnp_net_cfg.ROT_TYPE == "ego_lie_vec":
            dataset_dict["ego_lie_vec"] = lie_algebra.rot_to_lie_vec_direct(
                torch.as_tensor(pose[:3, :3].astype("float32")[None])
            )[0]
        elif pnp_net_cfg.ROT_TYPE == "allo_lie_vec":
            dataset_dict["allo_lie_vec"] = lie_algebra.rot_to_lie_vec_direct(
                torch.as_tensor(allo_pose[:3, :3].astype("float32"))[None]
            )[0]
        else:
            raise ValueError(f"Unknown rot type: {pnp_net_cfg.ROT_TYPE}")
        dataset_dict["ego_rot"] = torch.as_tensor(pose[:3, :3].astype("float32"))
        dataset_dict["trans"] = torch.as_tensor(inst_infos["trans"].astype("float32"))

        model_points = torch.as_tensor(self._get_model_points(dataset_name)[roi_cls].astype("float32"))
        dataset_dict["model_points"] = model_points
        dataset_dict["sym_info"] = self._get_sym_infos(dataset_name)[roi_cls]
        if str(roi_cls + 1) not in self.model_infos:
            print(f"model_info not found for roi_cls: {roi_cls + 1}")
        model_info = copy.deepcopy(self.model_infos[str(roi_cls + 1)])
        if r_head_cfg.NUM_REGIONS > 1 and fps_points is not None:
            model_info["fps_points"] = np.asarray(fps_points, dtype=np.float32)
        dataset_dict["model_info"] = model_info

        dataset_dict["roi_img"] = torch.as_tensor(roi_img.astype("float32")).contiguous()
        dataset_dict["roi_coord_2d"] = torch.as_tensor(roi_coord_2d.astype("float32")).contiguous()

        dataset_dict["roi_mask_trunc"] = torch.as_tensor(roi_mask_trunc.astype("float32")).contiguous()
        dataset_dict["roi_mask_visib"] = torch.as_tensor(roi_mask_visib.astype("float32")).contiguous()
        dataset_dict["roi_mask_obj"] = torch.as_tensor(roi_mask_obj.astype("float32")).contiguous()

        dataset_dict["bbox_center"] = torch.as_tensor(bbox_center, dtype=torch.float32)
        dataset_dict["scale"] = scale
        dataset_dict["bbox"] = anno["bbox"]  # NOTE: original bbox
        dataset_dict["roi_wh"] = torch.as_tensor(np.array([bw, bh], dtype=np.float32))
        dataset_dict["resize_ratio"] = resize_ratio = out_res / scale
        z_ratio = inst_infos["trans"][2] / resize_ratio
        obj_center = anno["centroid_2d"]
        delta_c = obj_center - bbox_center
        dataset_dict["trans_ratio"] = torch.as_tensor([delta_c[0] / bw, delta_c[1] / bh, z_ratio]).to(torch.float32)
        # patch mask add for token retrieval
        dataset_dict["patch_mask"] = torch.as_tensor(patch_mask.astype("bool")).contiguous()

        # FOR DEBUGGING THE POSE ======================================================
        # import sys
        # sys.path.append("/home/renchengwei/GDR-Net/tools")
        # from visualize_3d_bbox import visualize_example
        # vis_output_dir = "/home/renchengwei/GDR-Net/debug"
        # input_images = dataset_dict["input_images"]
        # mean = torch.tensor([0.485, 0.456, 0.406], dtype=input_images.dtype).to(input_images.device)
        # std = torch.tensor([0.229, 0.224, 0.225], dtype=input_images.dtype).to(input_images.device)
        # mean = mean.unsqueeze(-1).unsqueeze(-1)
        # std = std.unsqueeze(-1).unsqueeze(-1)
        # input_images = (input_images * std + mean) * 255.0
        # input_images = input_images.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    
        # # visualize the input image, GT rot, GT trans, pred rot, pred trans
        # obj_cls = dataset_dict["roi_cls"] + 1 # since the obj_cls is 0-based index
        # # obj_cls_name = data_ref.id2obj[obj_cls]
        # dset_meta = MetadataCatalog.get(dataset_name)
        # ref_key = dset_meta.ref_key
        # data_ref = ref.__dict__[ref_key]
        # model_info = data_ref.get_models_info()[str(obj_cls)]
        # mins = [model_info["min_x"] / 1000.0, model_info["min_y"] / 1000.0, model_info["min_z"] / 1000.0]
        # sizes = [model_info["size_x"] / 1000.0, model_info["size_y"] / 1000.0, model_info["size_z"] / 1000.0]
        # size = mins + sizes
        # RT = np.concatenate([dataset_dict["ego_rot"].detach().cpu().numpy(), dataset_dict["trans"].detach().cpu().numpy()[..., None]], axis=1)
        # visualize_example(K=None, image=input_images, RT=RT, size=size, output_path=osp.join(vis_output_dir, f"gt.png"))
        # visualize_example(K=None, image=input_images, RT=RT, size=size, output_path=osp.join(vis_output_dir, f"pred.png"))
        # import pdb; pdb.set_trace()
        # print(f"scene_im_id: {dataset_dict['scene_im_id']}, file_name: {dataset_dict['file_name']}, trans: {dataset_dict['trans'].detach().cpu().numpy()}")


        return dataset_dict

    def smooth_xyz(self, xyz):
        """smooth the edge areas to reduce noise."""
        xyz = np.asarray(xyz, np.float32)
        xyz_blur = cv2.medianBlur(xyz, 3)
        edges = get_edge(xyz)
        xyz[edges != 0] = xyz_blur[edges != 0]
        return xyz

    def __getitem__(self, idx):
        if self.split != "train":
            dataset_dict_list = self._get_sample_dict(idx)
            # read_data returns a list of single-instance dicts per view
            all_views_inst_dicts = []  # [view0_inst_list, view1_inst_list, ...]
            for dataset_dict in dataset_dict_list:
                inst_dicts = self.read_data(dataset_dict)  # list of dicts
                all_views_inst_dicts.append(inst_dicts)

            num_inst = len(all_views_inst_dicts[0])
            N_views = len(all_views_inst_dicts)

            # Group by instance, stack across views (same as training)
            per_inst_stacked = []
            for inst_i in range(num_inst):
                views_for_inst = [all_views_inst_dicts[v][inst_i] for v in range(N_views)]
                per_inst_stacked.append(self.stack_processed_data(views_for_inst))

            # Batch across instances → (B, N, ...)
            return self.batch_test_instances(per_inst_stacked)

        while True:  # return valid data for train
            dataset_dict_list = self._get_sample_dict(idx)
            # Sample one contiguous training window before any heavy per-view processing
            # so data_fetch cost scales with window length instead of full scene_frame_num.
            if isinstance(dataset_dict_list, (list, tuple)) and len(dataset_dict_list) > 0:
                num_views_total = len(dataset_dict_list)
                num_context_views = int(self.cfg.MODEL.CDPN.PNP_NET.get("TRAIN_NUM_CONTEXT_VIEWS", 3))
                num_target_views = int(self.cfg.MODEL.CDPN.PNP_NET.get("TRAIN_NUM_TARGET_VIEWS", 3))
                window_len = min(max(1, num_context_views + num_target_views), num_views_total)
                if num_views_total > window_len:
                    start_idx = np.random.randint(0, num_views_total - window_len + 1)
                    end_idx = start_idx + window_len
                    dataset_dict_list = dataset_dict_list[start_idx:end_idx]
            processed_data_list = []
            valid_flag = True
            for dataset_dict in dataset_dict_list:
                processed_data = self.read_data(dataset_dict)
                if processed_data is None:
                    valid_flag = False
                    break
                processed_data_list.append(processed_data)
            if valid_flag:
                return self.stack_processed_data(processed_data_list)
            idx = self._rand_another(idx)
            
    def stack_processed_data(self, data):
        output_dict = {}
        device = data[0]["roi_img"].device
        output_dict["roi_img"] = torch.stack([d["roi_img"] for d in data], dim=0).to(device, non_blocking=True)
        # output_dict["roi_cls"] = torch.tensor([d["roi_cls"] for d in data], dtype=torch.long).to(device, non_blocking=True)
        # the roi_cls is the same for all the data, so we can use the first one
        output_dict["roi_cls"] = data[0]["roi_cls"]
        if "roi_coord_2d" in data[0]:
            output_dict["roi_coord_2d"] = torch.stack([d["roi_coord_2d"] for d in data], dim=0).to(
                device=device, non_blocking=True
            )
    
        output_dict["cam"] = torch.stack([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
        output_dict["bbox_center"] = torch.stack([d["bbox_center"] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        if "scale" in data[0]:
            output_dict["scale"] = torch.stack([torch.tensor([d["scale"]], dtype=torch.float32) for d in data], dim=0).to(device, non_blocking=True)
        output_dict["roi_wh"] = torch.stack([d["roi_wh"] for d in data], dim=0).to(device, non_blocking=True)
        # resize_ratio is a scalar float per view, stack to (N_views,)
        output_dict["resize_ratio"] = torch.tensor([d["resize_ratio"] for d in data]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        output_dict["roi_extent"] = torch.stack([d["roi_extent"] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
    
        if "trans_ratio" in data[0]:
            output_dict["trans_ratio"] = torch.stack([d["trans_ratio"] for d in data], dim=0).to(device, non_blocking=True)
        if "input_images" in data[0]:
            output_dict["input_images"] = torch.stack([d["input_images"] for d in data], dim=0).to(device, non_blocking=True)
        if "input_depths" in data[0]:
            output_dict["input_depths"] = torch.stack([d["input_depths"] for d in data], dim=0).to(
                device=device, non_blocking=True
            )
        if "input_obj_masks" in data[0]:
            output_dict["input_obj_masks"] = torch.stack([d["input_obj_masks"] for d in data], dim=0).to(
                device=device, non_blocking=True
            )
        if "noisy_obj_mask" in data[0]:
            output_dict["noisy_obj_mask"] = torch.stack([d["noisy_obj_mask"] for d in data], dim=0).to(
                device=device, non_blocking=True
            )
        # patch mask add for token retrieval
        if "patch_mask" in data[0]:
            output_dict["patch_mask"] = torch.stack([d["patch_mask"] for d in data], dim=0).to(device, non_blocking=True)
        # yapf: disable
        for key in [
            "roi_xyz",
            "roi_xyz_bin",
            "roi_mask_trunc",
            "roi_mask_visib",
            "roi_mask_obj",
            "roi_region",
            "ego_quat",
            "allo_quat",
            "ego_rot6d",
            "allo_rot6d",
            "ego_rot",
            "trans",
            "model_points",
        ]:
            if key in data[0]:
                if key in ["roi_region"]:
                    dtype = torch.long
                else:
                    dtype = torch.float32
                output_dict[key] = torch.stack([d[key] for d in data], dim=0).to(
                    device=device, dtype=dtype, non_blocking=True
                )
        # yapf: enable

        # here are same for all the data, so we can use the first one
        for key in [
            "sym_info",
            "inst_id",
            "dataset_name",
            "file_name",
            "depth_file",
            "height",
            "width",
            "image_id",
            "scene_im_id",
            "depth_factor",
            "img_type",
        ]:
            if key in data[0]:
                output_dict[key] = [d[key] for d in data]
                
        if "model_info" in data[0]:
            output_dict["model_info"] = data[0]["model_info"]

        # score: float per view → tensor (N_views,)
        if "score" in data[0]:
            if isinstance(data[0]["score"], torch.Tensor):
                output_dict["score"] = torch.stack([d["score"] for d in data], dim=0).to(device, non_blocking=True)
            else:
                output_dict["score"] = torch.tensor([d["score"] for d in data]).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

        return output_dict

    def batch_test_instances(self, inst_dicts):
        """Batch M per-instance stacked dicts into one dict with instance dimension.

        Input: list of M dicts, each from stack_processed_data with shape (N, ...)
        Output: one dict with shape (B, N, ...) where B = M = num_instances
        """
        output = {}
        device = inst_dicts[0]["roi_img"].device

        # Tensor fields: stack (N, ...) → (B, N, ...)
        for key in ["roi_img", "roi_coord_2d", "cam", "bbox_center", "roi_wh",
                     "roi_extent", "patch_mask", "input_images", "input_depths", "input_obj_masks", "resize_ratio",
                     "scale"]:
            if key in inst_dicts[0]:
                output[key] = torch.stack([d[key] for d in inst_dicts], dim=0).to(device)

        for key in ["ego_rot", "trans"]:
            if key in inst_dicts[0]:
                output[key] = torch.stack([d[key] for d in inst_dicts], dim=0).to(device)

        # roi_cls: int per instance → tensor (B,)
        if isinstance(inst_dicts[0]["roi_cls"], int):
            output["roi_cls"] = torch.tensor([d["roi_cls"] for d in inst_dicts], dtype=torch.long).to(device)
        else:
            output["roi_cls"] = torch.stack([d["roi_cls"] for d in inst_dicts], dim=0).to(device)

        # score: tensor (N,) per instance → (B, N)
        if "score" in inst_dicts[0]:
            output["score"] = torch.stack([d["score"] for d in inst_dicts], dim=0).to(device)

        # List-type fields: collect into list of lists
        for key in ["file_name", "scene_im_id", "inst_id", "dataset_name",
                     "height", "width"]:
            if key in inst_dicts[0]:
                output[key] = [d[key] for d in inst_dicts]

        # model_info: list of dicts/values
        if "model_info" in inst_dicts[0]:
            output["model_info"] = [d["model_info"] for d in inst_dicts]

        return output

    @staticmethod
    def augment_mask_noise(mask, cfg):
        """Apply random noise to a binary mask to simulate inference artifacts.

        Each noise type fires independently with its own probability.
        Multiple types can stack on one sample.

        Args:
            mask: (H, W) float32, binary mask (0 or 1).
            cfg: config with INPUT.MASK_NOISE_* keys.
        Returns:
            noisy_mask: (H, W) float32, augmented binary mask.
        """
        noisy = mask.copy()
        H, W = noisy.shape[:2]

        # --- 1. blob: random false-positive ellipses near the object ---
        if np.random.rand() < cfg.INPUT.MASK_NOISE_BLOB_PROB:
            ys, xs = np.where(noisy > 0.5)
            if len(ys) > 0:
                x1m, x2m = xs.min(), xs.max()
                y1m, y2m = ys.min(), ys.max()
                bw = max(x2m - x1m + 1, 1)
                bh = max(y2m - y1m + 1, 1)
                cx0 = 0.5 * (x1m + x2m)
                cy0 = 0.5 * (y1m + y2m)
                side = max(bw, bh)
                half_side = max(int(np.ceil(0.75 * side)), 1)
                ex1 = max(int(np.floor(cx0 - half_side)), 0)
                ex2 = min(int(np.ceil(cx0 + half_side)), W - 1)
                ey1 = max(int(np.floor(cy0 - half_side)), 0)
                ey2 = min(int(np.ceil(cy0 + half_side)), H - 1)
                ring_mask = np.zeros((H, W), dtype=np.uint8)
                ring_mask[ey1 : ey2 + 1, ex1 : ex2 + 1] = 1
                ring_mask[y1m : y2m + 1, x1m : x2m + 1] = 0
                cand_ys, cand_xs = np.where(ring_mask > 0)
                if len(cand_ys) == 0:
                    cand_ys, cand_xs = np.where(ring_mask[ey1 : ey2 + 1, ex1 : ex2 + 1] >= 0)
                    cand_ys = cand_ys + ey1
                    cand_xs = cand_xs + ex1
                n_blobs = np.random.randint(1, 3)  # 1-2 blobs
                for _ in range(n_blobs):
                    idx = np.random.randint(0, len(cand_ys))
                    cx = int(cand_xs[idx])
                    cy = int(cand_ys[idx])
                    rx = np.random.randint(5, 31)
                    ry = np.random.randint(5, 31)
                    angle = np.random.randint(0, 360)
                    cv2.ellipse(noisy, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)

        # --- 2. spur: thin connected line from boundary ---
        if np.random.rand() < cfg.INPUT.MASK_NOISE_SPUR_PROB:
            mask_u8 = (noisy > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                # pick the largest contour
                contour = max(contours, key=cv2.contourArea)
                if len(contour) > 0:
                    n_spurs = np.random.randint(1, 3)
                    for _ in range(n_spurs):
                        idx = np.random.randint(0, len(contour))
                        pt = contour[idx][0]
                        angle = np.random.uniform(0, 2 * np.pi)
                        length = np.random.randint(20, 81)
                        thickness = np.random.randint(2, 6)
                        end_pt = (
                            int(pt[0] + length * np.cos(angle)),
                            int(pt[1] + length * np.sin(angle)),
                        )
                        cv2.line(noisy, tuple(pt), end_pt, 1.0, thickness)

        # --- 3. erode: boundary erosion (false negatives) ---
        if np.random.rand() < cfg.INPUT.MASK_NOISE_ERODE_PROB:
            k = np.random.choice([3, 5, 7])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            noisy = cv2.erode(noisy, kernel, iterations=1)

        # --- 4. dilate: boundary dilation ---
        if np.random.rand() < cfg.INPUT.MASK_NOISE_DILATE_PROB:
            k = np.random.choice([3, 5, 7, 9, 11])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            noisy = cv2.dilate(noisy, kernel, iterations=1)

        # --- 5. dropout: random rectangular holes near the object boundary ---
        if np.random.rand() < cfg.INPUT.MASK_NOISE_DROPOUT_PROB:
            mask_u8 = (noisy > 0.5).astype(np.uint8)
            ys, xs = np.where(mask_u8 > 0)
            if len(ys) > 10:
                x1m, x2m = xs.min(), xs.max()
                y1m, y2m = ys.min(), ys.max()
                bw = max(x2m - x1m + 1, 1)
                bh = max(y2m - y1m + 1, 1)
                boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                boundary_band = mask_u8 - cv2.erode(mask_u8, boundary_kernel, iterations=1)
                bys, bxs = np.where(boundary_band > 0)
                if len(bys) > 0:
                    n_rects = np.random.randint(1, 3)
                    for _ in range(n_rects):
                        rw_min = max(1, int(bw * 0.1))
                        rw_max = max(rw_min, int(bw * 0.3))
                        rh_min = max(1, int(bh * 0.1))
                        rh_max = max(rh_min, int(bh * 0.3))
                        rw = np.random.randint(rw_min, rw_max + 1)
                        rh = np.random.randint(rh_min, rh_max + 1)
                        idx = np.random.randint(0, len(bys))
                        px = int(bxs[idx])
                        py = int(bys[idx])
                        rx = int(np.clip(px - np.random.randint(0, rw + 1), 0, max(W - rw, 0)))
                        ry = int(np.clip(py - np.random.randint(0, rh + 1), 0, max(H - rh, 0)))
                        noisy[ry : ry + rh, rx : rx + rw] = 0.0

        # re-binarize
        noisy = (noisy > 0.5).astype(np.float32)
        return noisy

    def aug_bbox(self, cfg, bbox_xyxy, im_H, im_W, patch_size):
        x1, y1, x2, y2 = bbox_xyxy.copy()
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bh = y2 - y1
        bw = x2 - x1
        if cfg.INPUT.DZI_TYPE.lower() == "uniform":
            scale_ratio = 1 + cfg.INPUT.DZI_SCALE_RATIO * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
            shift_ratio = cfg.INPUT.DZI_SHIFT_RATIO * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
            bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
            scale = max(y2 - y1, x2 - x1) * scale_ratio * cfg.INPUT.DZI_PAD_SCALE
            # import ipdb; ipdb.set_trace()
        elif cfg.INPUT.DZI_TYPE.lower() == "roi10d":
            # shift (x1,y1), (x2,y2) by 15% in each direction
            _a = -0.15
            _b = 0.15
            x1 += bw * (np.random.rand() * (_b - _a) + _a)
            x2 += bw * (np.random.rand() * (_b - _a) + _a)
            y1 += bh * (np.random.rand() * (_b - _a) + _a)
            y2 += bh * (np.random.rand() * (_b - _a) + _a)
            x1 = min(max(x1, 0), im_W)
            x2 = min(max(x1, 0), im_W)
            y1 = min(max(y1, 0), im_H)
            y2 = min(max(y2, 0), im_H)
            bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
            scale = max(y2 - y1, x2 - x1) * cfg.INPUT.DZI_PAD_SCALE
        elif cfg.INPUT.DZI_TYPE.lower() == "truncnorm":
            raise NotImplementedError("DZI truncnorm not implemented yet.")
        else:
            bbox_center = np.array([cx, cy])  # (w/2, h/2)
            scale = max(y2 - y1, x2 - x1)
        scale = min(scale, max(im_H, im_W)) * 1.0
        
        if cfg.INPUT.DZI_PATCH_GRID_ADSORPTION:
            # re-shift the bbox's four corners to the patch's corner
            xmin = bbox_center[0] - scale / 2
            ymin = bbox_center[1] - scale / 2
            xmax = bbox_center[0] + scale / 2
            ymax = bbox_center[1] + scale / 2
            
            # convert to patch coordinates
            px_min = np.floor(xmin / patch_size)
            py_min = np.floor(ymin / patch_size)
            px_max = np.ceil(xmax / patch_size)
            py_max = np.ceil(ymax / patch_size)
            
            # OPTIONAL: enforce square & even number of patches
            pw = px_max - px_min
            ph = py_max - py_min
            p = max(pw, ph)
            
            # make patch count even → center exactly on patch grid
            if p % 2 == 1:
                p += 1
            
            cx_p = (px_min + px_max) / 2
            cy_p = (py_min + py_max) / 2
            px_min = cx_p - p / 2
            px_max = cx_p + p / 2
            py_min = cy_p - p / 2
            py_max = cy_p + p / 2
            
            # back to pixel coordinates
            xmin = px_min * patch_size
            xmax = px_max * patch_size
            ymin = py_min * patch_size
            ymax = py_max * patch_size
            
            # update center & scale
            bbox_center = np.array([
                (xmin + xmax) / 2,
                (ymin + ymax) / 2
            ], dtype=np.float32)
            
            scale = (xmax - xmin)  # guaranteed multiple of patch_size
            
        return bbox_center, scale



def bbox_to_patch_mask(center, scale, patch_size, Ph, Pw):
    """
    center: (cx, cy) in pixel
    scale: float, bbox size in pixel (square)
    patch_size: int
    Ph, Pw: patch grid size (H, W)

    return:
        mask: (Ph, Pw) bool array
    """
    cx, cy = center
    half = scale / 2

    # patch index range (inclusive)
    xmin = math.floor((cx - half) / patch_size) * patch_size
    ymin = math.floor((cy - half) / patch_size) * patch_size
    xmax = math.ceil((cx + half) / patch_size) * patch_size
    ymax = math.ceil((cy + half) / patch_size) * patch_size
    
    px_min = xmin // patch_size
    px_max = xmax // patch_size - 1
    py_min = ymin // patch_size
    py_max = ymax // patch_size - 1

    # clamp to valid grid
    px_min = max(px_min, 0)
    py_min = max(py_min, 0)
    px_max = min(px_max, Pw - 1)
    py_max = min(py_max, Ph - 1)

    mask = np.zeros((Ph, Pw), dtype=bool)
    mask[py_min:py_max + 1, px_min:px_max + 1] = True

    return mask






def build_gdrn_train_loader(cfg, dataset_names):
    """A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts = get_detection_dataset_dicts(
        dataset_names,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    # TODO(rcw): here we disable the invalid instance filtering, maybe we should add it back
    # dataset_dicts = filter_invalid_in_dataset_dicts(dataset_dicts, visib_thr=cfg.DATALOADER.FILTER_VISIB_THR)
    


    dataset = GDRN_DatasetFromList(cfg, split="train", lst=dataset_dicts, copy=False)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return my_build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=bool(cfg.DATALOADER.get("PIN_MEMORY", True)),
    )


def build_gdrn_test_loader(cfg, dataset_name, train_objs=None):
    """Similar to `build_detection_train_loader`. But this function uses the
    given `dataset_name` argument (instead of the names in cfg), and uses batch
    size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # load test detection results
    if cfg.MODEL.LOAD_DETS_TEST:
        det_files = cfg.DATASETS.DET_FILES_TEST
        assert len(cfg.DATASETS.TEST) == len(det_files)
        load_detections_into_dataset(
            dataset_name,
            dataset_dicts,
            det_file=det_files[cfg.DATASETS.TEST.index(dataset_name)],
            top_k_per_obj=cfg.DATASETS.DET_TOPK_PER_OBJ,
            score_thr=cfg.DATASETS.DET_THR,
            train_objs=train_objs,
        )
        if cfg.DATALOADER.FILTER_EMPTY_DETS:
            dataset_dicts = filter_empty_dets(dataset_dicts)

    dataset = GDRN_DatasetFromList(cfg, split="test", lst=dataset_dicts, flatten=False)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    # Horovod: limit # of CPU threads to be used per worker.
    # if num_workers > 0:
    #     torch.set_num_threads(num_workers)

    kwargs = {"num_workers": num_workers}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
    # if (num_workers > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=trivial_batch_collator, **kwargs
    )
    return data_loader
