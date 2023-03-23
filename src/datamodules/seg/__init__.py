# """
# Dataset setup and loaders
# """

# # from .cityscapes import CityscapesDataSet
# # import .gta5.GTA5DataSet as GTA5DataSet
# # from .bdd100k import BDD100KDataSet
# # from .mapillary import MapillaryDataSet
# # from .synthia import SynthiaDataSet
# # from .acdc import ACDCDataSet
# # from .zurich_night import Zurich_night_DataSet
# # from .cross_city import CrosscityDataSet
# # from .KITTI import KITTIDataSet
# from torch.utils import data
# import torchvision.transforms as standard_transforms
# # import transforms.transforms as extended_transforms
# # import transforms.joint_transforms as joint_transforms
# from torchvision import transforms
# import numpy as np
# num_classes = 19
# ignore_label = 255

# def get_input_transforms(args):
#     """
#     Get input transforms
#     Args:
#         args: input config arguments
#         dataset: dataset class object

#     return: train_input_transform, val_input_transform
#     """

#     # Image appearance transformations
#     train_input_transform = []
#     val_input_transform = []
#     if args.color_aug > 0.0:
#         train_input_transform += [standard_transforms.RandomApply([
#             standard_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)]

#     if args.bblur:
#         train_input_transform += [extended_transforms.RandomBilateralBlur()]
#     elif args.gblur:
#         train_input_transform += [extended_transforms.RandomGaussianBlur()]

#     train_input_transform += [
#                                 standard_transforms.ToTensor()
#     ]
#     val_input_transform += [
#                             standard_transforms.ToTensor()
#     ]
#     train_input_transform = standard_transforms.Compose(train_input_transform)
#     val_input_transform = standard_transforms.Compose(val_input_transform)

#     return train_input_transform, val_input_transform

# def get_target_transforms(args, dataset):
#     """
#     Get target transforms
#     Args:
#         args: input config arguments
#         dataset: dataset class object

#     return: target_transform, target_train_transform, target_aux_train_transform
#     """

#     target_transform = extended_transforms.MaskToTensor()
#     if args.jointwtborder:
#         target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(
#                 dataset.ignore_label, dataset.num_classes)
#     else:
#         target_train_transform = extended_transforms.MaskToTensor()

#     target_aux_train_transform = extended_transforms.MaskToTensor()

#     return target_transform, target_train_transform, target_aux_train_transform

# def setup_loaders(args, dataset, cfg):
#     """
#     Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
#     input: argument passed by the user
#     return:  training data loader, validation data loader loader,  train_set
#     """

#     data_dict = {'cityscapes':CityscapesDataSet,
#                  'mapillary':MapillaryDataSet,
#                  'bdd100k':BDD100KDataSet,
#                  'synthia':SynthiaDataSet,
#                  'gtav':GTA5DataSet}

#     data_path_dict = {'cityscapes': cfg.DATASET.CITYSCAPES_DIR,
#                       'mapillary': cfg.DATASET.MAPILLARY_DIR ,
#                       'bdd100k': cfg.DATASET.BDD_DIR,
#                       'synthia': cfg.DATASET.SYNTHIA_DIR,
#                       'gtav': cfg.DATASET.GTAV_DIR,
#                       'acdc': cfg.DATASET.ACDC_DIR}

#     if dataset == 'cityscapes':
#         test_dataset = data_dict[dataset](
#             root=data_path_dict[dataset],
#             list_path=cfg.DATA_LIST_TARGET,
#             set=cfg.TEST.SET_TARGET,
#             info_path=cfg.TEST.INFO_TARGET,
#             crop_size=cfg.TEST.INPUT_SIZE_TARGET,
#             mean=np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32),
#             labels_size=cfg.TEST.OUTPUT_SIZE_TARGET
#         )

#         test_loader = data.DataLoader(test_dataset,
#                                       batch_size=cfg.TEST.BATCH_SIZE_TARGET,
#                                       num_workers=cfg.NUM_WORKERS,
#                                       shuffle=False,
#                                       pin_memory=True)
#     elif dataset == 'bdd100k':
#         val_input_transform = [
#             standard_transforms.ToTensor()
#         ]
#         val_input_transform = standard_transforms.Compose(val_input_transform)
#         target_transform = extended_transforms.MaskToTensor()
#         test_dataset = BDD100KDataSet('val', 0,
#                                       transform=val_input_transform,
#                                       target_transform=target_transform,
#                                       cv_split=0,
#                                       image_in=False)
#         test_loader = data.DataLoader(test_dataset,
#                                       batch_size=cfg.TEST.BATCH_SIZE_TARGET,
#                                       num_workers=cfg.NUM_WORKERS,
#                                       shuffle=False,
#                                       pin_memory=True)
#     elif dataset == 'mapillary':
#         train_input_transform, val_input_transform = get_input_transforms(args)
#         target_transform, target_train_transform, target_aux_train_transform = get_target_transforms(args, dataset)
#         eval_size = 1024
#         val_joint_transform_list = [
#             joint_transforms.ResizeHeight(eval_size),
#             joint_transforms.CenterCropPad(eval_size)]
#         test_dataset = mapillary.MapillaryDataSet('semantic', 'val',
#                                       joint_transform_list=val_joint_transform_list,
#                                       transform=val_input_transform,
#                                       target_transform=target_transform,
#                                       test=False)

#         test_loader = data.DataLoader(test_dataset,
#                                       batch_size=cfg.TEST.BATCH_SIZE_TARGET,
#                                       num_workers=cfg.NUM_WORKERS,
#                                       shuffle=False,
#                                       pin_memory=True)

#     elif dataset == 'acdc':
#         input_transform = transforms.Compose([
#             transforms.ToTensor(),
#             #transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
#         ])
#         test_dataset = acdc.ACDCDataSet('acdc', split='test', mode='testval', transform=input_transform)

#         test_loader = data.DataLoader(test_dataset,
#                                       batch_size=cfg.TEST.BATCH_SIZE_TARGET,
#                                       num_workers=cfg.NUM_WORKERS,
#                                       shuffle=False,
#                                       pin_memory=True)

#     elif dataset == 'Zurich_night':
#         zn_data_dir = '/home/hxx/data/davian/segmentation/zurich_night'
#         zn_data_list = '/home/hxx/Documents/my_code/Dynamic_Norm/advent/dataset/zurich_night_list/zurich_val.txt'
#         set = 'val'
#         test_dataset = Zurich_night_DataSet(
#             root='/home/hxx/data/davian/segmentation/zurich_night/',
#             list_path='/home/hxx/Documents/my_code/Dynamic_Norm/advent/dataset/zurich_night_list/zurich_val.txt',
#             set='val',
#             info_path='/home/hxx/Documents/my_code/Dynamic_Norm/advent/dataset/bdd100k_list/info.json',
#             crop_size=cfg.TEST.INPUT_SIZE_TARGET,
#             mean=np.array([0.485, 0.456, 0.406],dtype=np.float32)*255,
#             labels_size=cfg.TEST.OUTPUT_SIZE_TARGET
#         )

#         test_loader = data.DataLoader(test_dataset,
#                                       batch_size=cfg.TEST.BATCH_SIZE_TARGET,
#                                       num_workers=cfg.NUM_WORKERS,
#                                       shuffle=False,
#                                       pin_memory=True)

#     elif dataset == 'cross_city':
#         test_dataset = CrosscityDataSet(
#             root='/home/hxx/data/davian/segmentation/NTHU/',
#             list_path='/home/hxx/Documents/my_code/Dynamic_Norm/advent/dataset/cross_city_list/test.txt',
#             set=args.testing_city, #select testing city
#             crop_size=cfg.TEST.INPUT_SIZE_TARGET,
#             mean=None,
#             labels_size=cfg.TEST.OUTPUT_SIZE_TARGET
#         )

#         test_loader = data.DataLoader(test_dataset,
#                                       batch_size=cfg.TEST.BATCH_SIZE_TARGET,
#                                       num_workers=cfg.NUM_WORKERS,
#                                       shuffle=False,
#                                       pin_memory=True)

#     elif dataset == 'KITTI':
#         test_dataset = KITTIDataSet(
#             root='/home/hxx/data/davian/segmentation/KITTI/',
#             list_path='/home/hxx/Documents/my_code/Dynamic_Norm/advent/dataset/KITTI_list/train.txt',
#             set='train', #select testing city
#             crop_size=cfg.TEST.INPUT_SIZE_TARGET,
#             mean=None,
#             labels_size=cfg.TEST.OUTPUT_SIZE_TARGET
#         )

#         test_loader = data.DataLoader(test_dataset,
#                                       batch_size=cfg.TEST.BATCH_SIZE_TARGET,
#                                       num_workers=cfg.NUM_WORKERS,
#                                       shuffle=False,
#                                       pin_memory=True)


#     return test_loader
