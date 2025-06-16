import dolfin
import numpy as np
import gmsh # For using the Gmsh Python API
import os # For file operations like removing temp files
import uuid # For unique temporary file names

# --- Default Geometry Parameters for Pipe ---
DEFAULT_PIPE_LENGTH = 3.0
DEFAULT_PIPE_HEIGHT = 1.0
DEFAULT_OBSTACLE_CENTER_X_FACTOR = 0.3
DEFAULT_OBSTACLE_CENTER_Y_FACTOR = 0.5
DEFAULT_OBSTACLE_RADIUS_FACTOR = 0.15 # Factor of PIPE_HEIGHT
DEFAULT_GMSH_MESH_SIZE_FACTOR = 0.1 # Factor for characteristic length (smaller is finer)

def create_square_mesh(nx, ny):
    """
    Creates a simple FEniCS UnitSquareMesh.
    """
    print(f"Creating Square Mesh: nx={nx}, ny={ny}")
    mesh = dolfin.UnitSquareMesh(nx, ny)
    dimensions = {
        "width": 1.0,
        "height": 1.0,
        "center_x": 0.5,
        "center_y": 0.5,
        "type": "square"
    }
    return mesh, dimensions

def create_pipe_with_obstacle_mesh_gmsh(
    pipe_length=DEFAULT_PIPE_LENGTH,
    pipe_height=DEFAULT_PIPE_HEIGHT,
    obstacle_cx_factor=DEFAULT_OBSTACLE_CENTER_X_FACTOR,
    obstacle_cy_factor=DEFAULT_OBSTACLE_CENTER_Y_FACTOR,
    obstacle_r_factor=DEFAULT_OBSTACLE_RADIUS_FACTOR,
    mesh_size_factor=DEFAULT_GMSH_MESH_SIZE_FACTOR):
    """
    Creates a FEniCS mesh of a pipe with a circular obstacle using the Gmsh Python API.

    Args:
        pipe_length (float): Length of the pipe.
        pipe_height (float): Height of the pipe.
        obstacle_cx_factor (float): Obstacle center x-coordinate as a factor of pipe_length.
        obstacle_cy_factor (float): Obstacle center y-coordinate as a factor of pipe_height.
        obstacle_r_factor (float): Obstacle radius as a factor of pipe_height.
        mesh_size_factor (float): Factor to determine Gmsh characteristic mesh length.
                                  (e.g., mesh_size = mesh_size_factor * pipe_height).

    Returns:
        dolfin.Mesh: The generated FEniCS mesh.
        dict: A dictionary containing key dimensions.
    """
    print(f"Creating Pipe with Obstacle Mesh (using Gmsh Python API)...")
    
    obstacle_cx = pipe_length * obstacle_cx_factor
    obstacle_cy = pipe_height * obstacle_cy_factor
    obstacle_r = pipe_height * obstacle_r_factor
    
    # Characteristic length for meshing (smaller means finer mesh)
    # We can base it on the obstacle radius or pipe height
    char_len = min(obstacle_r if obstacle_r > 1e-9 else pipe_height, pipe_height) * mesh_size_factor
    if char_len < 1e-9 : char_len = 0.05 # A small default if calculated is too small

    # Temporary file for Gmsh output
    # Create a unique filename to avoid conflicts if multiple instances run
    temp_msh_filename = f"temp_mesh_{uuid.uuid4()}.msh"

    mesh = None
    actual_obstacle_radius = 0.0 # Will be updated if obstacle is successfully created

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0) # Suppress Gmsh messages in terminal
        gmsh.model.add("pipe_with_obstacle")

        # Define outer pipe rectangle
        # Points: (x, y, z, characteristic_length)
        p1 = gmsh.model.geo.addPoint(0, 0, 0, char_len)
        p2 = gmsh.model.geo.addPoint(pipe_length, 0, 0, char_len)
        p3 = gmsh.model.geo.addPoint(pipe_length, pipe_height, 0, char_len)
        p4 = gmsh.model.geo.addPoint(0, pipe_height, 0, char_len)

        # Lines for the rectangle
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        # Curve loop and surface for the rectangle
        cl_rect = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        s_rect = gmsh.model.geo.addPlaneSurface([cl_rect])

        # Define circular obstacle if radius is valid
        # Basic check: ensure obstacle is reasonably within the pipe and not too small
        can_create_obstacle = (
            obstacle_r > 1e-3 and # Obstacle radius should be significant
            obstacle_cx - obstacle_r > 0.001 * pipe_length and
            obstacle_cx + obstacle_r < 0.999 * pipe_length and
            obstacle_cy - obstacle_r > 0.001 * pipe_height and
            obstacle_cy + obstacle_r < 0.999 * pipe_height
        )

        if can_create_obstacle:
            pc = gmsh.model.geo.addPoint(obstacle_cx, obstacle_cy, 0, char_len * 0.5) # finer mesh near circle
            # Points on the circle circumference (relative to center pc)
            p_c1 = gmsh.model.geo.addPoint(obstacle_cx + obstacle_r, obstacle_cy, 0, char_len * 0.5)
            p_c2 = gmsh.model.geo.addPoint(obstacle_cx - obstacle_r, obstacle_cy, 0, char_len * 0.5)
            # Circle arcs: addCircleArc(start_point_tag, center_point_tag, end_point_tag)
            c_arc1 = gmsh.model.geo.addCircleArc(p_c1, pc, p_c2)
            c_arc2 = gmsh.model.geo.addCircleArc(p_c2, pc, p_c1)
            
            cl_circle = gmsh.model.geo.addCurveLoop([c_arc1, c_arc2])
            s_circle = gmsh.model.geo.addPlaneSurface([cl_circle])

            # Perform boolean cut (subtract circle from rectangle)
            # outTags = gmsh.model.geo.cut(objectTags, toolTags, removeObject=True, removeTool=True)
            # objectTags: [(dim, tag)], toolTags: [(dim, tag)]
            # Here, s_rect is dim 2, tag will be 1. s_circle is dim 2, tag will be 2.
            # (Tags are usually sequential starting from 1 for each dimension)
            cut_result = gmsh.model.geo.cut([(2, s_rect)], [(2, s_circle)], removeObject=True, removeTool=True)
            
            # cut_result is a list of (dim, tag) pairs for the resulting entities,
            # and a map of parent-child relationships. The first element of cut_result[0]
            # should be the tag of the resulting surface with the hole.
            if not cut_result or not cut_result[0]:
                print("Warning: Gmsh boolean cut operation failed. Meshing pipe without obstacle.")
                # No need to do anything here, s_rect (without hole) will be meshed.
            else:
                actual_obstacle_radius = obstacle_r
                print("Gmsh boolean cut for obstacle successful.")
        else:
            print("Warning: Obstacle parameters invalid or obstacle too close to boundary. Meshing pipe without obstacle.")
            # s_rect (without hole) will be meshed.

        gmsh.model.geo.synchronize()

        # Define physical groups (optional but good practice)
        # The tag of the final surface needs to be identified.
        # If cut happened, it's in cut_result[0][0][1]. If no cut, it's s_rect.
        surfaces = gmsh.model.getEntities(2) # Get all 2D entities
        if surfaces: # Should always be at least one surface
            final_surface_tag = surfaces[-1][1] # Assume the last one is the one we want (or the one from cut)
            gmsh.model.addPhysicalGroup(2, [final_surface_tag], name="fluid_domain")
        else:
            raise RuntimeError("Gmsh: No surface defined for meshing.")


        # Generate 2D mesh
        gmsh.model.mesh.generate(2)

        # Set MSH file version to 2.2 for FEniCS 2019.1.0 compatibility
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(temp_msh_filename)
        print(f"Gmsh mesh written to {temp_msh_filename}")

    except Exception as e:
        print(f"Error during Gmsh geometry definition or meshing: {e}")
        # Fallback dimensions in case of Gmsh error before mesh creation
        dimensions = {
            "width": pipe_length, "height": pipe_height, "center_x": pipe_length / 2.0,
            "center_y": pipe_height / 2.0, "obstacle_radius": 0.0,
            "obstacle_center_x": obstacle_cx, "obstacle_center_y": obstacle_cy,
            "type": "pipe_gmsh_error"
        }
        return dolfin.UnitSquareMesh(5,5), dimensions # Return a dummy mesh
    finally:
        gmsh.finalize() # Always finalize Gmsh

    # Load mesh into FEniCS
    try:
        if os.path.exists(temp_msh_filename) and os.path.getsize(temp_msh_filename) > 0:
            mesh = dolfin.Mesh()
            # Use XDMFFile for robust reading if available, or Mesh directly for .msh
            # FEniCS 2019.1.0 should be able to read MSH v2.2 directly with dolfin.Mesh
            # However, sometimes dolfin-convert to XML was more stable.
            # Let's try direct loading first.
            dolfin_mesh_temp = dolfin.Mesh(temp_msh_filename) # FEniCS tries to read it
            mesh = dolfin_mesh_temp # If successful
            print(f"FEniCS mesh loaded from {temp_msh_filename} with {mesh.num_cells()} cells.")
            if mesh.num_cells() == 0:
                print("Warning: Loaded FEniCS mesh is empty (0 cells).")
        else:
            print(f"Error: Gmsh output file {temp_msh_filename} not found or is empty.")
            raise RuntimeError("Gmsh .msh file missing or empty.")
            
    except Exception as e:
        print(f"Error loading Gmsh .msh file into FEniCS: {e}")
        dimensions = {
            "width": pipe_length, "height": pipe_height, "center_x": pipe_length / 2.0,
            "center_y": pipe_height / 2.0, "obstacle_radius": 0.0,
            "obstacle_center_x": obstacle_cx, "obstacle_center_y": obstacle_cy,
            "type": "pipe_gmsh_fenics_load_error"
        }
        return dolfin.UnitSquareMesh(5,5), dimensions # Return a dummy mesh
    finally:
        # Clean up the temporary .msh file
        if os.path.exists(temp_msh_filename):
            try:
                os.remove(temp_msh_filename)
                print(f"Temporary file {temp_msh_filename} removed.")
            except OSError as e:
                print(f"Error removing temporary file {temp_msh_filename}: {e}")
    
    if mesh is None or mesh.num_cells() == 0: # Final check
        print("Critical Error: Gmsh mesh generation and loading resulted in an invalid or empty FEniCS mesh.")
        dimensions = {"type": "fallback_square_due_to_error_gmsh"}
        return dolfin.UnitSquareMesh(5,5), dimensions

    dimensions = {
        "width": pipe_length,
        "height": pipe_height,
        "center_x": pipe_length / 2.0,
        "center_y": pipe_height / 2.0,
        "obstacle_radius": actual_obstacle_radius,
        "obstacle_center_x": obstacle_cx,
        "obstacle_center_y": obstacle_cy,
        "type": "pipe_gmsh"
    }
    return mesh, dimensions

if __name__ == '__main__':
    print("Testing mesh_generators.py with Gmsh Python API...")
    import matplotlib.pyplot as plt

    # Test Square Mesh (remains unchanged)
    sq_mesh, sq_dims = create_square_mesh(nx=10, ny=10)
    print(f"Square mesh created with {sq_mesh.num_cells()} cells. Dimensions: {sq_dims}")
    if sq_mesh.num_cells() > 0:
        plt.figure(figsize=(6,6))
        dolfin.plot(sq_mesh, title="Test Square Mesh")

    # Test Pipe with Obstacle Mesh using Gmsh
    print("\nTesting Pipe with Obstacle (Gmsh)...")
    pipe_mesh_gmsh, pipe_dims_gmsh = create_pipe_with_obstacle_mesh_gmsh(
        pipe_length=3.0, pipe_height=1.0,
        obstacle_cx_factor=0.3, obstacle_cy_factor=0.5,
        obstacle_r_factor=0.15,
        mesh_size_factor=0.1 # Adjust for finer/coarser mesh
    )
    print(f"Pipe mesh (Gmsh) created with {pipe_mesh_gmsh.num_cells()} cells. Obstacle radius: {pipe_dims_gmsh.get('obstacle_radius')}")
    if pipe_mesh_gmsh.num_cells() > 0 and pipe_dims_gmsh.get("type") != "fallback_square_due_to_error_gmsh":
        plt.figure(figsize=(10, (10 * pipe_dims_gmsh.get('height',1) / pipe_dims_gmsh.get('width',3)) if pipe_dims_gmsh.get('width',3) > 0 else 4 ))
        dolfin.plot(pipe_mesh_gmsh, title="Test Pipe with Obstacle (Gmsh API)")
        plt.gca().set_aspect('equal', adjustable='box')

    # Test pipe without a significant obstacle (small radius)
    print("\nTesting Pipe with Tiny Obstacle (Gmsh)...")
    pipe_mesh_tiny_obs_gmsh, pipe_dims_tiny_obs_gmsh = create_pipe_with_obstacle_mesh_gmsh(
        obstacle_r_factor=0.001, # Effectively no obstacle
        mesh_size_factor=0.2
    )
    print(f"Pipe mesh (Gmsh, tiny obs) created with {pipe_mesh_tiny_obs_gmsh.num_cells()} cells. Obstacle radius: {pipe_dims_tiny_obs_gmsh.get('obstacle_radius')}")
    if pipe_mesh_tiny_obs_gmsh.num_cells() > 0 and pipe_dims_tiny_obs_gmsh.get("type") != "fallback_square_due_to_error_gmsh":
        plt.figure(figsize=(10, (10 * pipe_dims_tiny_obs_gmsh.get('height',1) / pipe_dims_tiny_obs_gmsh.get('width',3))  if pipe_dims_tiny_obs_gmsh.get('width',3) > 0 else 4 ))
        dolfin.plot(pipe_mesh_tiny_obs_gmsh, title="Test Pipe - Tiny Obstacle (Gmsh API)")
        plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()
    print("Mesh generation tests complete.")
