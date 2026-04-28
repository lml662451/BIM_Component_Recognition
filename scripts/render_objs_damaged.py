import bpy
import os
import random
import pandas as pd
from mathutils import Vector, Matrix
import logging
from logging.handlers import RotatingFileHandler
import time

def setup_logger(name='blender_render_logger', log_dir='blender_render_logs', max_bytes=10*1024*1024, backup_count=5):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"render_damage_{time.strftime('%Y%m%d')}.log")
    
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

random.seed(42)

def reset_scene():
    try:
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj)
        for coll in bpy.data.collections:
            if coll.name != "Master Collection":
                bpy.data.collections.remove(coll)
        for mat in bpy.data.materials:
            bpy.data.materials.remove(mat)
        for light in bpy.data.lights:
            bpy.data.lights.remove(light)
        for cam in bpy.data.cameras:
            bpy.data.cameras.remove(cam)
    except Exception as e:
        logger.error(f"重置场景时发生错误: {e}", exc_info=True)

def setup_render_settings():
    try:
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.render.resolution_x = 640
        scene.render.resolution_y = 480
        scene.render.resolution_percentage = 100
        scene.cycles.samples = 16
        scene.cycles.device = 'CPU'
        scene.render.film_transparent = False
    except Exception as e:
        logger.error(f"配置渲染设置时发生错误: {e}", exc_info=True)

def setup_scene_environment():
    try:
        cam = bpy.data.cameras.new(name="Camera")
        cam_obj = bpy.data.objects.new("Camera", cam)
        bpy.context.collection.objects.link(cam_obj)
        cam_obj.location = (5.0, -5.0, 4.0)
        cam_obj.rotation_euler = (1.1, 0.0, 0.785)
        bpy.context.scene.camera = cam_obj
        
        sun = bpy.data.lights.new(name="Sun", type='SUN')
        sun_obj = bpy.data.objects.new("Sun", sun)
        bpy.context.collection.objects.link(sun_obj)
        sun_obj.location = (10.0, 10.0, 10.0)
        sun.energy = 5.0
        
        area = bpy.data.lights.new(name="Area", type='AREA')
        area_obj = bpy.data.objects.new("Area", area)
        bpy.context.collection.objects.link(area_obj)
        area_obj.location = (-5.0, 5.0, 8.0)
        area.energy = 100.0
    except Exception as e:
        logger.error(f"配置场景环境时发生错误: {e}", exc_info=True)

def import_and_scale_model(obj_path):
    try:
        if not os.path.exists(obj_path):
            return None
        
        if bpy.app.version >= (3, 0, 0):
            bpy.ops.wm.obj_import(filepath=obj_path)
        else:
            bpy.ops.import_scene.obj(filepath=obj_path)
        
        if not bpy.context.selected_objects:
            return None
        
        obj = bpy.context.selected_objects[0]
        obj.location = (0, 0, 0)
        dim = max(obj.dimensions)
        if dim > 0:
            scale_val = 2.0 / dim
            obj.scale = (scale_val, scale_val, scale_val)
        bpy.ops.object.transform_apply(scale=True)
        return obj
    except Exception as e:
        logger.error(f"导入/缩放模型 {obj_path} 时发生错误: {e}", exc_info=True)
        return None

def create_realistic_hole_damage(obj):
    try:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='OBJECT')
        
        hole_count = random.randint(1, 3)
        for _ in range(hole_count):
            face = random.choice([f for f in obj.data.polygons if f.area > 0.01])
            center = obj.matrix_world @ face.center
            normal = face.normal @ obj.matrix_world.to_3x3()
            normal.normalize()
            
            base_radius = random.uniform(0.08, 0.18)
            radius_variation = random.uniform(0.8, 1.2)
            
            bpy.ops.mesh.primitive_icosphere_add(
                subdivisions=2,
                radius=base_radius * radius_variation,
                location=center + normal * random.uniform(0.1, 0.3)
            )
            cutter = bpy.context.active_object
            cutter.rotation_euler = (
                random.uniform(0, 3.14),
                random.uniform(0, 3.14),
                random.uniform(0, 3.14)
            )
            
            bool_mod = obj.modifiers.new(name=f"HoleBool_{_}", type='BOOLEAN')
            bool_mod.operation = 'DIFFERENCE'
            bool_mod.object = cutter
            bool_mod.solver = 'FAST'
            
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=bool_mod.name)
            bpy.data.objects.remove(cutter)
        
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.vertices_smooth(factor=0.1, repeat=2)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception as e:
        logger.error(f"创建真实感孔洞破损时发生错误: {e}", exc_info=True)
    return obj

def create_realistic_missing_part_damage(obj):
    try:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='OBJECT')
        
        bbox = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
        min_z = min(v.z for v in bbox)
        max_z = max(v.z for v in bbox)
        min_x = min(v.x for v in bbox)
        max_x = max(v.x for v in bbox)
        
        cut_z = min_z + (max_z - min_z) * random.uniform(0.2, 0.5)
        cut_x = min_x + (max_x - min_x) * random.uniform(0.3, 0.7)
        
        bpy.ops.mesh.primitive_cube_add(size=5, location=(cut_x, 0, cut_z))
        cutter = bpy.context.active_object
        cutter.scale = (
            random.uniform(8, 12),
            random.uniform(8, 12),
            random.uniform(0.8, 1.2)
        )
        cutter.rotation_euler = (
            0,
            0,
            random.uniform(-0.2, 0.2)
        )
        
        bool_mod = obj.modifiers.new(name="CutBool", type='BOOLEAN')
        bool_mod.operation = 'DIFFERENCE'
        bool_mod.object = cutter
        bool_mod.solver = 'FAST'
        
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="CutBool")
        bpy.data.objects.remove(cutter)
        
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.bevel(offset=0.01, offset_type='OFFSET', segments=2)
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception as e:
        logger.error(f"创建真实感缺失部分破损时发生错误: {e}", exc_info=True)
    return obj

def create_realistic_spalling_damage(obj):
    try:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='OBJECT')
        
        spall_count = random.randint(3, 8)
        for _ in range(spall_count):
            verts = [v for v in obj.data.vertices if obj.matrix_world @ v.co]
            if not verts:
                continue
            vert = random.choice(verts)
            center = obj.matrix_world @ vert.co
            
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=random.uniform(0.05, 0.12),
                location=center + Vector((
                    random.uniform(-0.05, 0.05),
                    random.uniform(-0.05, 0.05),
                    random.uniform(-0.05, 0.05)
                ))
            )
            cutter = bpy.context.active_object
            
            bool_mod = obj.modifiers.new(name=f"SpallBool_{_}", type='BOOLEAN')
            bool_mod.operation = 'DIFFERENCE'
            bool_mod.object = cutter
            bool_mod.solver = 'FAST'
            
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=bool_mod.name)
            bpy.data.objects.remove(cutter)
        
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.vertices_smooth(factor=0.05, repeat=1)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception as e:
        logger.error(f"创建真实感剥落破损时发生错误: {e}", exc_info=True)
    return obj

def add_surface_scratches(obj):
    try:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='OBJECT')
        
        scratch_mod = obj.modifiers.new(name="Scratches", type='DISPLACE')
        noise_tex = bpy.data.textures.new(name="ScratchNoise", type='NOISE')
        noise_tex.noise_type = 'VORONOI'
        noise_tex.scale = random.uniform(5, 15)
        noise_tex.intensity = random.uniform(0.01, 0.03)
        
        scratch_mod.texture = noise_tex
        scratch_mod.strength = random.uniform(0.005, 0.015)
        
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="Scratches")
        
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception as e:
        pass
    return obj

def apply_realistic_damage(obj):
    if obj is None:
        return None
    funcs = [create_realistic_hole_damage, create_realistic_missing_part_damage, create_realistic_spalling_damage]
    f = random.choice(funcs)
    obj = f(obj)
    
    if random.random() > 0.3:
        obj = add_surface_scratches(obj)
    
    return obj

def render_damaged_model(obj_path, output_path):
    try:
        reset_scene()
        setup_render_settings()
        setup_scene_environment()
        obj = import_and_scale_model(obj_path)
        obj = apply_realistic_damage(obj)
        if obj is None:
            return
        
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
    except Exception as e:
        logger.error(f"渲染模型 {obj_path} 时发生错误: {e}", exc_info=True)

def batch_render_damaged(csv_path, output_root):
    try:
        if not os.path.exists(csv_path):
            return
        
        os.makedirs(output_root, exist_ok=True)
        df = pd.read_csv(csv_path)
        
        success_count = 0
        fail_count = 0
        
        for idx, row in df.iterrows():
            try:
                ct = row['component_type']
                pt = row['project_type']
                op = row['instance_path']
                
                if not os.path.exists(op):
                    fail_count += 1
                    continue
                
                obj_name = os.path.basename(op).split('.')[0]
                iid = obj_name.split('_')[-1]
                save_dir = os.path.join(output_root, ct, pt)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{iid}.png")
                
                render_damaged_model(op, save_path)
                success_count += 1
            except Exception as e:
                logger.error(f"处理第 {idx+1} 条记录时发生错误: {e}", exc_info=True)
                fail_count += 1
    except Exception as e:
        logger.critical(f"批量渲染主流程发生严重错误: {e}", exc_info=True)

if __name__ == "__main__":
    CSV_PATH = r"D:\BIM_Component_Recognition\metadata\relabel_kept.csv"
    OUTPUT_DIR = r"D:\BIM_Component_Recognition\datasets\damaged"
    
    batch_render_damaged(CSV_PATH, OUTPUT_DIR)