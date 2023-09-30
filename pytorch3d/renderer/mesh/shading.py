# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple
import math
import torch
from pytorch3d.ops import interpolate_face_attributes
import torch.nn.functional as F
from .textures import TexturesVertex


def _apply_lighting(
    points, normals, lights, cameras, materials, diffNormals = None,
    shininess = None, highlight = 'phong', shadows = None, albedo=False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        points: torch tensor of shape (N, ..., 3) or (P, 3).
        normals: torch tensor of shape (N, ..., 3) or (P, 3)
        lights: instance of the Lights class.
        cameras: instance of the Cameras class.
        materials: instance of the Materials class.

    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    if diffNormals is None:
        diffNormals = normals

    if shininess is None:
        shininess = materials.shininess

    if shadows is None:
        # If no shadows are given, illuminate all 
        shadows = torch.ones_like(shininess)

    light_diffuse = lights.diffuse(normals=diffNormals, points=points)
    light_specular = lights.specular(
        normals=normals,
        points=points,
        camera_position=cameras.get_camera_center(),
        shininess=shininess,
        highlight=highlight,
    )

    if albedo:
        materials.ambient_color = 1
        materials.diffuse_color = 1
        materials.specular_color = 1

    ambient_color = materials.ambient_color * lights.ambient_color
    diffuse_color = materials.diffuse_color * light_diffuse * shadows
    specular_color = materials.specular_color * light_specular * shadows

    if normals.dim() == 2 and points.dim() == 2:
        # If given packed inputs remove batch dim in output.
        return (
            ambient_color.squeeze(),
            diffuse_color.squeeze(),
            specular_color.squeeze(),
        )
    

    if ambient_color.ndim != diffuse_color.ndim:
        # Reshape from (N, 3) to have dimensions compatible with
        # diffushighlightnt_color, diffuse_color, specular_color
        ambient_color = ambient_color[:, None, None, None, :]
    
    return ambient_color, diffuse_color, specular_color

def gaussian_kernel(size, sigma=20., dim=2, channels=3):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.
    # TODO: don't call every time.

    kernel_size = 2*size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    return kernel.float()

def gaussian_blur(x, size):
    x = x.permute(0, 3, 1, 2)
    kernel = gaussian_kernel(size=size).to(x.device)
    kernel_size = 2*size + 1

    padding = int((kernel_size - 1) / 2)
    x = F.pad(x, (padding, padding, padding, padding), mode='replicate')

    x = F.conv2d(x, kernel, groups=3)

    return x.permute(0, 2, 3, 1)

def _apply_subsurface_scattering(diff_term, translucency=None, kernel_size=21):
    """
    Apply simply subsurface scattering using the method from Matrix Reloaded (Borshukov and Lewis, 2003).
    First blur the diffuse lighting compoment,
    weight it per channel following the NVIDIA tutorial (d'Eon, 2007),
    and then combine it with the diffuse term.
    Args:
        diff_term: The calculated diffuse term from phong shading (N, H, W, K, 3)
        translucency: optional UV map for translucency. using uniform 1s if not given (N, H, W, K, 3)
        blur_kernel: blur kernel in pixels, corresponging to 1.4mm 
    Returns:
        diff_term_sss _apply_lighting(N, H, W, K, 3)
    """

    # Set Params

    W_C = [0.4, 0.1, 0.035] # Based on NVIDIA SIGGRAPH Tut, averaged as single filter
    W_E = 1.3 # Energy preservation weight

    # Define Translucency based on the weights per channel
    if translucency is None:
        return diff_term
    
    elif translucency.shape != diff_term.shape:
        if len(translucency.shape) == 4:
            translucency = translucency.repeat(1, 1, 1, 3)
        if len(translucency.shape) == 5:
            translucency = translucency.repeat(1, 1, 1, 3, 1)

    translucency[:,:,:,:,0] *= W_C[0]
    translucency[:,:,:,:,1] *= W_C[1]
    translucency[:,:,:,:,2] *= W_C[2]

    # Compute Blur
    diff_term_blur = W_E * gaussian_blur(diff_term.mean(3).clone(), kernel_size).unsqueeze(-2)

    # Weight each channel
    diff_term = translucency * diff_term_blur + (1-translucency) * diff_term

    return diff_term

def phong_shading(
    meshes, fragments, lights, cameras, materials, texels
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )

    colors = (ambient + diffuse) * texels + specular
    return colors

def multi_blinnphong_shading(
    meshes, fragments, lights, cameras, materials, texel_group, highlight='blinn_phong', normal_space='object'
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.
    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texel_group: texture per pixel of shape (N, H, W, K, 3) for all maps
    Returns:
        colors: (N, H, W, K, 3)
    """

    if texel_group['diffAlbedo'] is not None:
        diffAlbedo = texel_group['diffAlbedo']
    else:
        diffAlbedo = texel_group['texture']

    if texel_group['specAlbedo'] is not None:
        specAlbedo = texel_group['specAlbedo']
    else:
        specAlbedo = torch.ones_like(diffAlbedo)[..., 0:1]

    shininess = texel_group['shininess'] if 'shininess' in texel_group.keys() else torch.ones_like(specAlbedo).mean(-3)
    shadows = texel_group['shadows'] if 'shadows' in texel_group.keys() else torch.ones_like(specAlbedo)
    translucency = texel_group['translucency']

    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )

    if normal_space == 'tangent' or texel_group['diffNormals'] is None or texel_group['specNormals'] is None:
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals)   

    if texel_group['diffNormals'] is not None:
        diffNormals = texel_group['diffNormals'] * 2. - 1.
    else:
        diffNormals = pixel_normals

    if texel_group['specNormals'] is not None:
        specNormals = texel_group['specNormals'] * 2. - 1.
    else:
        specNormals = pixel_normals     

    if normal_space == 'tangent':
        specNormals = _transform_tangent_normals(pixel_normals, specNormals)

    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, specNormals, lights, cameras, materials, diffNormals, shininess, highlight, shadows
    )

    diff_term = (ambient + diffuse) * diffAlbedo
    spec_term = specular * specAlbedo

    diff_term = _apply_subsurface_scattering(diff_term, translucency=translucency)
    
    colors = diff_term + spec_term
    colors = torch.clamp(colors, 0, 1)

    return colors

def _transform_tangent_normals(object_normals, tangent_normals):
    """
    Transform the tangent normals uv map (Darboux) to object space using the object normals
    """

    object_normals[..., 0] += tangent_normals[..., 0]
    object_normals[..., 1] += tangent_normals[..., 1]
    object_normals = F.normalize(object_normals, p=2, dim=-1, eps=1e-6)
    return object_normals

def multi_blinnphong_baking(
    meshes, lights, cameras, materials, texel_group, highlight
) -> torch.Tensor:
    """
    Apply per pixel shading. Compute the illumination for each pixel,
    directly in texture space, using the geometry in UV space as well.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.
    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization (included for consistency)
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texel_group: texture per pixel of shape (N, H, W, K, 3) for all maps
    Returns:
        colors: (N, H, W, K, 3)
    """

    diffAlbedo = texel_group['diffAlbedo']
    specAlbedo = texel_group['specAlbedo']
    diffNormals = texel_group['diffNormals'] * 2. - 1.
    specNormals = texel_group['specNormals'] * 2. - 1.
    shininess = texel_group['shininess'] 
    vertices_uvs = texel_group['vertices_uvs'] * 2. - 1.
    shadows = texel_group['shadows']

    # transfor geometry

    # Shade
    ambient, diffuse, specular = _apply_lighting(
        vertices_uvs, specNormals, lights, cameras, materials, diffNormals, shininess, highlight, shadows
    )

    # combine
    diff_term = (ambient + diffuse) * diffAlbedo
    spec_term = specular * specAlbedo

    color = diff_term + spec_term

    return colors

def multi_blinnphong_baking_tensor(
    meshes, lights, cameras, materials, texel_group, highlight
) -> torch.Tensor:
    """
    Apply per pixel shading. Compute the illumination for each pixel,
    directly in texture space, using the geometry in UV space as well.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.
    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization (included for consistency)
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texel_group: texture per pixel of shape (N, H, W, K, 3) for all maps
    Returns:
        colors: (N, H, W, K, 3)
    """

    diffAlbedo = texel_group['diffAlbedo']
    specAlbedo = texel_group['specAlbedo']
    diffNormals = texel_group['diffNormals']
    specNormals = texel_group['specNormals']
    shininess = texel_group['shininess'] 
    vertices_uvs = texel_group['vertices_uvs']
    shadows = texel_group['shadows']
    translucency = texel_group['translucency']

    # combine
    diff_term = (ambient + diffuse) * diffAlbedo
    spec_term = specular * specAlbedo

    diff_term = _apply_subsurface_scattering(diff_term, translucency=translucency)

    colors = diff_term + spec_term

    return colors


def gouraud_shading(meshes, fragments, lights, cameras, materials) -> torch.Tensor:
    """
    Apply per vertex shading. First compute the vertex illumination by applying
    ambient, diffuse and specular lighting. If vertex color is available,
    combine the ambient and diffuse vertex illumination with the vertex color
    and add the specular component to determine the vertex shaded color.
    Then interpolate the vertex shaded colors using the barycentric coordinates
    to get a color per pixel.

    Gouraud shading is only supported for meshes with texture type `TexturesVertex`.
    This is because the illumination is applied to the vertex colors.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties

    Returns:
        colors: (N, H, W, K, 3)
    """
    if not isinstance(meshes.textures, TexturesVertex):
        raise ValueError("Mesh textures must be an instance of TexturesVertex")

    faces = meshes.faces_packed()  # (F, 3)
    verts = meshes.verts_packed()  # (V, 3)
    verts_normals = meshes.verts_normals_packed()  # (V, 3)
    verts_colors = meshes.textures.verts_features_packed()  # (V, D)
    vert_to_mesh_idx = meshes.verts_packed_to_mesh_idx()

    # Format properties of lights and materials so they are compatible
    # with the packed representation of the vertices. This transforms
    # all tensor properties in the class from shape (N, ...) -> (V, ...) where
    # V is the number of packed vertices. If the number of meshes in the
    # batch is one then this is not necessary.
    if len(meshes) > 1:
        lights = lights.clone().gather_props(vert_to_mesh_idx)
        cameras = cameras.clone().gather_props(vert_to_mesh_idx)
        materials = materials.clone().gather_props(vert_to_mesh_idx)

    # Calculate the illumination at each vertex
    ambient, diffuse, specular = _apply_lighting(
        verts, verts_normals, lights, cameras, materials
    )

    verts_colors_shaded = verts_colors * (ambient + diffuse) + specular
    face_colors = verts_colors_shaded[faces]
    colors = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, face_colors
    )
    return colors


def flat_shading(meshes, fragments, lights, cameras, materials, texels) -> torch.Tensor:
    """
    Apply per face shading. Use the average face position and the face normals
    to compute the ambient, diffuse and specular lighting. Apply the ambient
    and diffuse color to the pixel color and add the specular component to
    determine the final pixel color.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    face_normals = meshes.faces_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    face_coords = faces_verts.mean(dim=-2)  # (F, 3, XYZ) mean xyz across verts

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = fragments.pix_to_face == -1
    pix_to_face = fragments.pix_to_face.clone()
    pix_to_face[mask] = 0

    N, H, W, K = pix_to_face.shape
    idx = pix_to_face.view(N * H * W * K, 1).expand(N * H * W * K, 3)

    # gather pixel coords
    pixel_coords = face_coords.gather(0, idx).view(N, H, W, K, 3)
    pixel_coords[mask] = 0.0
    # gather pixel normals
    pixel_normals = face_normals.gather(0, idx).view(N, H, W, K, 3)
    pixel_normals[mask] = 0.0

    # Calculate the illumination at each face
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )
    colors = (ambient + diffuse) * texels + specular
    return colors
