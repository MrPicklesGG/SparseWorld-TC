import os
import cv2
import argparse
import importlib
import os.path as osp
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from datetime import datetime
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from loaders.builder import build_dataloader
import time


classname_to_color = {  # RGB.
    0: (0, 0, 0),  # Black. noise
    1: (112, 128, 144),  # Slategrey barrier
    2: (220, 20, 60),  # Crimson bicycle
    3: (255, 127, 80),  # Orangered bus
    4: (255, 158, 0),  # Orange car
    5: (233, 150, 70),  # Darksalmon construction
    6: (255, 61, 99),  # Red motorcycle
    7: (0, 0, 230),  # Blue pedestrian
    8: (47, 79, 79),  # Darkslategrey trafficcone
    9: (255, 140, 0),  # Darkorange trailer
    10: (255, 99, 71),  # Tomato truck
    11: (0, 207, 191),  # nuTonomy green driveable_surface
    12: (175, 0, 75),  # flat other
    13: (75, 0, 75),  # sidewalk
    14: (112, 180, 60),  # terrain
    15: (222, 184, 135),  # Burlywood mannade
    16: (0, 175, 0),  # Green vegetation
    17: (140, 140, 140),  # maybe unoccupied (255, 255, 255)?
}

palette = np.array([classname_to_color[i] for i in range(len(classname_to_color))])


def decode_points(points, pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]):
    points = points.copy()
    points[..., 0] = points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    points[..., 1] = points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    points[..., 2] = points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return points


def visualize_occ(x, y, z, labels, palette, voxel_size, classes, mode='cube', color=None, show=False):
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if palette.shape[1] == 3:
        palette = np.concatenate([palette, np.ones((palette.shape[0], 1)) * 255], axis=1)
    
    norm_labels = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
    
    colors = palette[labels.astype(int) % len(palette)] / 255.0
    
    scatter = ax.scatter(x, y, z, c=colors, s=voxel_size*10, marker='o', alpha=1.0)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_title('3D Occupancy Visualization')
    
    # ax.view_init(elev=30, azim=45)
    ax.view_init(elev=80, azim=180)
    
    if show:
        plt.show()
        plt.close()
        return None
    else:
        fig.canvas.draw()
        
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        plt.close()
        return img


def set_pv_camera_from_extrin_intrin(plotter, c2w, K, img_size, near=0.1, far=200.0):
    """
    T_cam_ego: 4x4, 把ego点变到camera
    K: 3x3 内参
    img_size: (H, W)
    """
    H, W = img_size
    T_cam_ego = np.linalg.inv(c2w)

    R = T_cam_ego[:3, :3]
    t = T_cam_ego[:3, 3]

    # 计算相机在ego坐标系中的位置与方向
    C = -R.T @ t                        # 相机中心在ego
    # C[1] = C[1] - 2  # pv offset
    x_axis = R.T @ np.array([1, 0, 0.0])  # 相机x轴在ego
    y_axis = R.T @ np.array([0, 1, 0.0])  # 相机y轴在ego（图像向下）
    z_axis = R.T @ np.array([0, 0, 1.0])  # 相机z轴在ego（朝前）

    cam = plotter.camera
    cam.position = C.tolist()
    cam.focal_point = (C + z_axis).tolist()   # 让相机朝z_axis看
    cam.up = (-y_axis).tolist()               # 使图像“向下”为正v（与常见相机一致）

    # cam.position = (0,0,0)
    # cam.focal_point = (0,0,1)
    # cam.up = (0,-1,0) # 如果你相机定义 y 向上，就改 (0,1,0)

    # 用 fy 和图像高度换算垂直视场角（单位度）
    fy = K[1, 1]
    fovy = 2.0 * np.degrees(np.arctan(0.5 * H / fy))
    cam.view_angle = float(fovy)

    cx, cy = K[0, 2], K[1, 2]
    wx = (W / 2.0 - cx) / (W / 2.0)
    wy = (cy - H / 2.0) / (H / 2.0)

    cam.clipping_range = (near, far)


def calculate_fut2cur():
    # 旋转角度 clockwise
    # theta = np.pi / 4
    theta = 0
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    R_cur2fut = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    # 平移向量
    t_cur2fut = np.array([-15, -5, 0])
    
    # cur2fut
    cur2fut = np.eye(4)
    cur2fut[:3, :3] = R_cur2fut
    cur2fut[:3, 3] = t_cur2fut
    
    # fut2cur
    R_fut2cur = R_cur2fut.T
    t_fut2cur = -R_fut2cur @ t_cur2fut
    
    fut2cur = np.eye(4)
    fut2cur[:3, :3] = R_fut2cur
    fut2cur[:3, 3] = t_fut2cur
    fut2cur = fut2cur.astype(np.float32)
    
    return fut2cur


# sudo apt install libgl1-mesa-glx xvfb
pv.start_xvfb()
def pyvista_vis_occ(x, y, z, labels, palette, voxel_size, classes, color=None, mode=None, c2w=None, K=None, img_size=None):
    
    point_colors = palette[labels]

    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.set_background('white')

    if mode == 'pv':
        set_pv_camera_from_extrin_intrin(plotter, c2w, K, img_size)
    elif mode == 'bev':
        plotter.view_xy()  # plotter.view_top() , need debug
    elif mode == 'ob':
        # observer view from rear-top
        position = (-80, 0, 50)
        focal_point = (0, 0, 0)
        view_up = (0, 0, 1)
        plotter.camera_position = [position, focal_point, view_up]       
    else:
        plotter.view_isometric()

    pts = np.stack((x, y, z), axis=-1)

    point_cloud = pv.PolyData(pts)
    point_cloud["colors"] = point_colors / 255.0

    cube = pv.Cube(center=(0, 0, 0), x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
    glyphs = point_cloud.glyph(scale=False, geom=cube)

    glyphs["colors"] = np.repeat(point_colors, cube.n_points, axis=0)

    plotter.add_mesh(glyphs, scalars="colors", rgb=True)

    origin_point = pv.PolyData([0, 0, 0])
    plotter.add_mesh(origin_point, color='yellow', point_size=20, render_points_as_spheres=True)

    # plotter.add_mesh(point_cloud, scalars="colors", rgb=True, point_size=4, render_points_as_spheres=True)

    
    # image = plotter.screenshot()
    # plotter.render() 
    plotter.show(auto_close=False)
    # time.sleep(5)
    image = plotter.screenshot()
    plotter.close()

    return image


def vis_iter(outs, fut2cur):
    mf_cls_scores = outs['all_cls_scores'][-1]  # last_layer
    mf_refine_pts = outs['all_refine_pts'][-1]
    frame_len = len(fut2cur)
    scene_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    for frame_i in range(frame_len):
        # cls_scores = outs['all_cls_scores'][-1].reshape(-1, 17)
        cls_scores = mf_cls_scores[frame_i].reshape(-1, 17)
        cls_scores = cls_scores.detach().cpu().numpy()
        label = cls_scores.argmax(axis=-1)
        # refine_pts = outs['all_refine_pts'][-1].reshape(-1, 3)
        refine_pts = mf_refine_pts[frame_i].reshape(-1, 3)
        refine_pts = refine_pts.detach().cpu().numpy()
        refine_pts = decode_points(refine_pts, scene_range)
        x, y, z = refine_pts[:, 0], refine_pts[:, 1], refine_pts[:, 2]
        
        img = pyvista_vis_occ(
            x, y, z,
            label,
            palette,
            0.4,
            list(classname_to_color.keys()),
            mode='ob'
            )
        cv2.imwrite(osp.join('./', f'frame{frame_i}th_result.jpg'), img[..., ::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--weights', required=True, help='Path to checkpoint')
    parser.add_argument('--vis-input', action='store_true', help='Visualize inputs')
    parser.add_argument('--vis-gt', action='store_true', help='Visualize ground-truths')
    parser.add_argument('--occ-root', default='data/nuscenes/gts/', help='Path to Occ3D')
    parser.add_argument('--save-dir', type=str, default='visualizations', help='Visualize results')
    parser.add_argument('--with-postprocess', action='store_true', help='Results post-processing')
    parser.add_argument('--override', nargs='+', action=DictAction, help='Override config')
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)
    
    run_name = osp.splitext(osp.split(args.config)[-1])[0]
    run_name += '_' + datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    work_dir = os.path.join(args.save_dir, run_name)
    if os.path.exists(work_dir):
        raise FileExistsError('Directory already exists')
    os.makedirs(work_dir, exist_ok=True)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    for p in cfgs.data.val.pipeline:
        if p['type'] == 'LoadMultiViewImageFromMultiSweeps':
            p['force_offline'] = True
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])

    load_checkpoint(model, args.weights, map_location='cuda', strict=False)
    model.eval()

    scene_range = cfgs.point_cloud_range
    voxel_size = cfgs.voxel_size
    W = int((scene_range[3] - scene_range[0]) / voxel_size[0])
    H = int((scene_range[4] - scene_range[1]) / voxel_size[1])
    Z = int((scene_range[5] - scene_range[2]) / voxel_size[2])

    x = (np.arange(0, W) + 0.5) * voxel_size[0] + scene_range[0]
    y = (np.arange(0, H) + 0.5) * voxel_size[1] + scene_range[1]
    z = (np.arange(0, Z) + 0.5) * voxel_size[2] + scene_range[2]
    xx = x[:, None, None].repeat(H, axis=1).repeat(Z, axis=2)
    yy = y[None, :, None].repeat(W, axis=0).repeat(Z, axis=2)
    zz = z[None, None, :].repeat(W, axis=0).repeat(H, axis=1)

    vis_ndarray = []
    # vis_list = [460, 517, 544, 575, 635, 695, 785, 812, 882, 972, 1020, 1140, 1314,
    #             1428, 1444, 1596, 1605, 1634, 1660, 1672, 1806, 1864, 1930, 2200, 2225]
    vis_id = [377]  # 377, 294
    with torch.no_grad():
        # for i, data in enumerate(val_loader):
        for i in vis_id:
            data = val_loader.dataset[i]
            # if i != vis_id:
                # continue
            # Visualize input data
            if args.vis_input:
                images = data['img'][0].data.cpu().numpy()[:6]
                filenames = data['img_metas'][0].data['filename'][:6]
                camera_names = [f.split('/')[-2] for f in filenames]

                camera_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
                ordered_images = [0 for _ in range(len(images))]
                for camera_name, img in zip(camera_names, images):
                    img = img.transpose(1, 2, 0).copy()
                    cv2.putText(img, camera_name, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    index = camera_order.index(camera_name)
                    ordered_images[index] = img
                
                H, W = ordered_images[0].shape[:2]
                ordered_images = np.stack(ordered_images, axis=0).reshape(2, 3, H, W, 3)
                ordered_images = ordered_images.transpose(0, 2, 1, 3, 4)
                ordered_images = ordered_images.reshape(2 * H, 3 * W, 3)
                cv2.imwrite(osp.join(work_dir, f'{i:0>6}_input.jpg'), ordered_images)

            if args.with_postprocess:
                result = model(return_loss=False, rescale=True, **data)
                label, pos = result[0]['sem_pred'], result[0]['occ_loc']
                x = xx[pos[:, 0], pos[:, 1], pos[:, 2]]
                y = yy[pos[:, 0], pos[:, 1], pos[:, 2]]
                z = zz[pos[:, 0], pos[:, 1], pos[:, 2]]
                img = visualize_occ(
                    x, y, z,
                    label,
                    palette,
                    0.4,
                    list(classname_to_color.keys()),
                    show=False)
                cv2.imwrite(osp.join(work_dir, f'{i:0>6}_result.jpg'), img[..., ::-1])
            else:
                img, img_metas = data['img'][0].data.unsqueeze(0), [data['img_metas'][0].data]
                
                fut_list = data['fut_list'][0]
                fut2cur = data['fut2cur'][0]
                # add offset('+': left, '-': right)
                # f2c = calculate_fut2cur()
                # fut2cur = [torch.from_numpy(f2c).to('cuda').unsqueeze(0)]
                fut2cur = [torch.from_numpy(f2c).to('cuda').unsqueeze(0) for f2c in fut2cur]

                # for pv_cam setting
                intrinsics = data['img_metas'][0].data['intrinsics']
                intrinsics = np.asarray(intrinsics).astype(np.float32)[:6, :3, :3]
                extrinsics = data['img_metas'][0].data['extrinsics']
                extrinsics = np.asarray(extrinsics).astype(np.float32)[:6]
                img_size = [256, 704]

                _model = model.module
                img_feats = _model.extract_feat(img=img.cuda(), img_metas=img_metas)
                '''
                img_feats[0].shape: torch.Size([1, 48, 256, 64, 176])
                img_feats[1].shape: torch.Size([1, 48, 256, 32, 88])
                img_feats[2].shape: torch.Size([1, 48, 256, 16, 44])
                img_feats[3].shape: torch.Size([1, 48, 256, 8, 22])
                '''
                outs = _model.pts_bbox_head(img_feats, img_metas, fut2cur, fut_list)
                '''
                outs = dict(init_points=init_points,     # [4, Q, 1, 3]
                            all_cls_scores=cls_scores,   # [4, Q, P, 17]
                            all_refine_pts=refine_pts,   # [4, Q, P, 3]
                            )
                '''
                mf_cls_scores = outs['all_cls_scores'][-1]  # last_layer
                mf_refine_pts = outs['all_refine_pts'][-1]
                frame_len = len(fut2cur)
                for frame_i in range(frame_len):
                    # cls_scores = outs['all_cls_scores'][-1].reshape(-1, 17)
                    cls_scores = mf_cls_scores[frame_i].reshape(-1, 17)
                    cls_scores = cls_scores.detach().cpu().numpy()
                    label = cls_scores.argmax(axis=-1)
                    # refine_pts = outs['all_refine_pts'][-1].reshape(-1, 3)
                    refine_pts = mf_refine_pts[frame_i].reshape(-1, 3)
                    refine_pts = refine_pts.detach().cpu().numpy()
                    refine_pts = decode_points(refine_pts, scene_range)
                    x, y, z = refine_pts[:, 0], refine_pts[:, 1], refine_pts[:, 2]
                    # img = visualize_occ(
                    #     x, y, z,
                    #     label,
                    #     palette,
                    #     0.4,
                    #     list(classname_to_color.keys()),
                    #     show=False
                    #     )
                    img = pyvista_vis_occ(
                        x, y, z,
                        label,
                        palette,
                        0.4,
                        list(classname_to_color.keys()),
                        mode='ob',
                        c2w=extrinsics[0],
                        K=intrinsics[0],
                        img_size=img_size
                        )
                    cv2.imwrite(osp.join(work_dir, f'{i:0>6}_frame{frame_i}th_result.jpg'), img[..., ::-1])
            
            # Visualize ground-truths
            if args.vis_gt:
                scene_name = val_dataset.data_infos[i]['scene_name']
                token = val_dataset.data_infos[i]['token']
                occ_file = osp.join(args.occ_root, scene_name, token, 'labels.npz')
                occ = np.load(occ_file)['semantics']
                x, y, z = xx[occ!=17], yy[occ!=17], zz[occ!=17]
                label = occ[occ!=17].astype(np.int64)

                img = pyvista_vis_occ(
                    x, y, z,
                    label,
                    palette,
                    0.4,
                    list(classname_to_color.keys()),
                    mode='ob',
                    c2w=extrinsics[0],
                    K=intrinsics[0],
                    img_size=img_size
                    )
                cv2.imwrite(osp.join(work_dir, f'{i:0>6}_gt.jpg'), img[..., ::-1])
