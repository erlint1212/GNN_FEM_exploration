# mesh_generators_2.py
import dolfin # Ensure dolfin is imported
import numpy as np
import gmsh
import os
import uuid
# import meshio # No longer strictly needed if only using direct .msh loading for Gmsh
# import h5py   # Explicitly removed

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
    Stores nx and ny in the dimensions dictionary.
    """
    print(f"Creating Square Mesh: nx={nx}, ny={ny}")
    # Ensure MPI communicator is handled for mesh creation if running in parallel,
    # though for this script, it's often run as a single process.
    # Using dolfin.MPI.comm_world by default if not specified.
    mesh = dolfin.UnitSquareMesh(nx, ny)
    dimensions = {
        "width": 1.0,
        "height": 1.0,
        "center_x": 0.5,
        "center_y": 0.5,
        "type": "square",
        "nx": nx, # Added
        "ny": ny  # Added
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
    Creates a FEniCS mesh of a pipe with a circular obstacle using the Gmsh Python API
    and loads the .msh file directly into FEniCS.
    """
    print(f"Creating Pipe with Obstacle Mesh (Gmsh API, direct .msh load)...")
    
    obstacle_cx = pipe_length * obstacle_cx_factor
    obstacle_cy = pipe_height * obstacle_cy_factor
    obstacle_r = pipe_height * obstacle_r_factor
    
    can_create_obstacle = (
        obstacle_r > 1e-3 and # Using a slightly larger threshold for "valid" radius
        obstacle_cx - obstacle_r > 0.001 * pipe_length and
        obstacle_cx + obstacle_r < 0.999 * pipe_length and
        obstacle_cy - obstacle_r > 0.001 * pipe_height and
        obstacle_cy + obstacle_r < 0.999 * pipe_height
    )

# Define characteristic lengths
    if can_create_obstacle:
        base_char_len_calc = obstacle_r 
        obstacle_points_char_len = min(base_char_len_calc, pipe_height) * mesh_size_factor * 0.5 
        domain_char_len = min(base_char_len_calc, pipe_height) * mesh_size_factor
    else:
        domain_char_len = pipe_height * mesh_size_factor
        obstacle_points_char_len = domain_char_len 

    domain_char_len = max(domain_char_len, 1e-4) 
    obstacle_points_char_len = max(obstacle_points_char_len, 1e-4)

    temp_msh_filename = f"temp_mesh_{uuid.uuid4().hex}.msh"
    fenics_mesh = None 
    actual_obstacle_radius = 0.0
    mesh_generated_successfully_by_gmsh = False

    # It's good practice to check if gmsh is initialized if running in scripts that might call this multiple times.
    # However, for a typical single run, direct initialize/finalize is fine.
    if not gmsh.isInitialized():
        gmsh.initialize()
    
    # Check if model already exists to avoid error, or add a new one.
    current_model_name = f"pipe_obstacle_{uuid.uuid4().hex}"
    try:
        gmsh.model.add(current_model_name)
    except Exception as e: # If add fails, it might be because a model is already there from a previous run not finalized
        print(f"Note: Gmsh model add failed (possibly already exists or other issue): {e}. Clearing and adding.")
        gmsh.clear() # Clear all models and start fresh
        gmsh.initialize() # Re-initialize after clear
        gmsh.model.add(current_model_name)


    try:
        gmsh.option.setNumber("General.Terminal", 0)
        p1 = gmsh.model.geo.addPoint(0, 0, 0, domain_char_len)
        p2 = gmsh.model.geo.addPoint(pipe_length, 0, 0, domain_char_len)
        p3 = gmsh.model.geo.addPoint(pipe_length, pipe_height, 0, domain_char_len)
        p4 = gmsh.model.geo.addPoint(0, pipe_height, 0, domain_char_len)
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        cl_rect = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        s_final_surface_tag = -1

        if can_create_obstacle:
            pc = gmsh.model.geo.addPoint(obstacle_cx, obstacle_cy, 0, obstacle_points_char_len)
            p_c1 = gmsh.model.geo.addPoint(obstacle_cx + obstacle_r, obstacle_cy, 0, obstacle_points_char_len)
            p_c2 = gmsh.model.geo.addPoint(obstacle_cx - obstacle_r, obstacle_cy, 0, obstacle_points_char_len)
            c_arc1 = gmsh.model.geo.addCircleArc(p_c1, pc, p_c2)
            c_arc2 = gmsh.model.geo.addCircleArc(p_c2, pc, p_c1)
            cl_circle = gmsh.model.geo.addCurveLoop([c_arc1, c_arc2])
            s_final_surface_tag = gmsh.model.geo.addPlaneSurface([cl_rect, cl_circle])
            actual_obstacle_radius = obstacle_r
            print("Gmsh surface with obstacle hole defined using addPlaneSurface.")
        else:
            print("Warning: Obstacle parameters invalid or obstacle too close to boundary. Meshing pipe without obstacle.")
            s_final_surface_tag = gmsh.model.geo.addPlaneSurface([cl_rect])
            actual_obstacle_radius = 0.0
        
        gmsh.model.geo.synchronize()
        if s_final_surface_tag == -1:
            raise RuntimeError("Gmsh: Final surface was not created.")
        
        # Physical group for the 2D surface
        gmsh.model.addPhysicalGroup(2, [s_final_surface_tag], name="fluid_domain")
        
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # Crucial for FEniCS compatibility
        gmsh.write(temp_msh_filename)
        print(f"Gmsh mesh written to {temp_msh_filename}")
        mesh_generated_successfully_by_gmsh = True

    except Exception as e_gmsh:
        print(f"Error during Gmsh (model: {gmsh.model.getCurrent()}) geometry/meshing: {e_gmsh}")
        import traceback
        traceback.print_exc()
    # No finally for gmsh.finalize() here, let it be handled outside or at end of script if multiple calls
    # For safety, if you call this function multiple times in a script, ensure gmsh.finalize() is handled appropriately.
    # If this is the only gmsh interaction per script run, finalize at the end of this function is okay.
    # However, if another function also uses gmsh, it might need gmsh to be initialized.
    # Let's keep finalize here assuming this function is self-contained for one mesh gen.
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


    # --- Load MSH directly into FEniCS, bypassing meshio for this step ---
    if mesh_generated_successfully_by_gmsh and os.path.exists(temp_msh_filename) and os.path.getsize(temp_msh_filename) > 0:
        try:
            print(f"Attempting to load GMSH MSH file directly with dolfin.Mesh: {temp_msh_filename}")
            # Pass MPI communicator for FEniCS mesh if running in parallel
            fenics_mesh = dolfin.Mesh(dolfin.MPI.comm_world, temp_msh_filename) # Direct load
            print(f"FEniCS mesh loaded directly from MSH with {fenics_mesh.num_cells()} cells and {fenics_mesh.num_vertices()} vertices.")
            if fenics_mesh.num_cells() == 0:
                print("Warning: Loaded FEniCS mesh via dolfin.Mesh from MSH is empty (0 cells).")
        
        except Exception as e_direct_load:
            print(f"Error during direct dolfin.Mesh MSH loading: {e_direct_load}")
            import traceback
            traceback.print_exc()
            fenics_mesh = None 
    else:
        if not mesh_generated_successfully_by_gmsh:
            print("Skipping FEniCS mesh loading because Gmsh did not complete successfully.")
        else:
            print(f"Skipping FEniCS mesh loading: MSH file '{temp_msh_filename}' not found or empty.")

    # Cleanup temporary MSH file
    if os.path.exists(temp_msh_filename):
        try:
            os.remove(temp_msh_filename)
            print(f"Temporary file {temp_msh_filename} removed.")
        except OSError as e_remove: # pragma: no cover
            print(f"Warning: Could not remove temporary file {temp_msh_filename}: {e_remove}")
            pass 

    # Fallback to dummy mesh if fenics_mesh is still None or empty
    if fenics_mesh is None or fenics_mesh.num_cells() == 0:
        print("FEniCS mesh is invalid or empty after Gmsh/direct MSH load. Returning a dummy mesh.")
        dimensions = {
            "width": pipe_length, "height": pipe_height, "center_x": pipe_length / 2.0,
            "center_y": pipe_height / 2.0, "obstacle_radius": 0.0, 
            "obstacle_center_x": obstacle_cx, "obstacle_center_y": obstacle_cy,
            "type": "pipe_fallback_dummy_direct_msh"
        }
        return dolfin.UnitSquareMesh(dolfin.MPI.comm_world, 5, 5), dimensions


    dimensions = {
        "width": pipe_length, "height": pipe_height, "center_x": pipe_length / 2.0,
        "center_y": pipe_height / 2.0, "obstacle_radius": actual_obstacle_radius,
        "obstacle_center_x": obstacle_cx, "obstacle_center_y": obstacle_cy,
        "type": "pipe_gmsh_direct_msh" # Updated type to reflect direct loading
    }
    return fenics_mesh, dimensions

if __name__ == '__main__':
    print("Testing mesh_generators_2.py with Gmsh Python API (direct .msh load)...")
    import matplotlib.pyplot as plt

    # Test Square Mesh
    print("\nTesting Square Mesh...")
    sq_mesh, sq_dims = create_square_mesh(nx=10, ny=10)
    print(f"Square mesh created with {sq_mesh.num_cells()} cells. Dimensions: {sq_dims}")
    if sq_mesh.num_cells() > 0:
        plt.figure(figsize=(6,6))
        dolfin.plot(sq_mesh)
        plt.title("Test Square Mesh")
        plt.savefig("test_square_mesh.png")
        plt.close()
        print("Saved test_square_mesh.png")


    # Test Pipe with Obstacle Mesh using Gmsh direct load
    print("\nTesting Pipe with Obstacle (Gmsh, direct .msh load)...")
    # Ensure Gmsh is finalized if it was used before or re-initialize cleanly
    if gmsh.isInitialized():
        gmsh.finalize()

    pipe_mesh_gmsh, pipe_dims_gmsh = create_pipe_with_obstacle_mesh_gmsh(
        pipe_length=3.0, pipe_height=1.0,
        obstacle_cx_factor=0.3, obstacle_cy_factor=0.5,
        obstacle_r_factor=0.15, # Valid obstacle
        mesh_size_factor=0.15 
    )
    print(f"Pipe mesh (Gmsh, direct) created with {pipe_mesh_gmsh.num_cells()} cells. Obstacle radius: {pipe_dims_gmsh.get('obstacle_radius')}")
    if pipe_mesh_gmsh.num_cells() > 0 and pipe_dims_gmsh.get("type") != "pipe_fallback_dummy_direct_msh":
        plt.figure(figsize=(10, (10 * pipe_dims_gmsh.get('height',1) / pipe_dims_gmsh.get('width',3)) if pipe_dims_gmsh.get('width',3) > 0 else 4 ))
        dolfin.plot(pipe_mesh_gmsh)
        plt.title("Test Pipe with Obstacle (Gmsh API, direct .msh load)")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("test_pipe_obstacle_direct_msh.png")
        plt.close()
        print("Saved test_pipe_obstacle_direct_msh.png")
    elif pipe_mesh_gmsh.num_cells() > 0 : # Fallback case might still plot a unit square
        plt.figure(figsize=(6,6))
        dolfin.plot(pipe_mesh_gmsh)
        plt.title("Test Pipe Fallback Mesh (Gmsh, direct .msh load)")
        plt.savefig("test_pipe_fallback_direct_msh.png")
        plt.close()
        print("Saved test_pipe_fallback_direct_msh.png")


    # Test pipe without a significant obstacle (should mesh the plain pipe)
    print("\nTesting Pipe without Obstacle (Gmsh, direct .msh load)...")
    if gmsh.isInitialized(): # Ensure clean state for Gmsh
        gmsh.finalize()
    pipe_mesh_no_obs_gmsh, pipe_dims_no_obs_gmsh = create_pipe_with_obstacle_mesh_gmsh(
        obstacle_r_factor=0.0001, # Effectively no obstacle
        mesh_size_factor=0.2
    )
    print(f"Pipe mesh (Gmsh, no obs, direct) created with {pipe_mesh_no_obs_gmsh.num_cells()} cells. Obstacle radius: {pipe_dims_no_obs_gmsh.get('obstacle_radius')}")
    if pipe_mesh_no_obs_gmsh.num_cells() > 0 and pipe_dims_no_obs_gmsh.get("type") != "pipe_fallback_dummy_direct_msh":
        plt.figure(figsize=(10, (10 * pipe_dims_no_obs_gmsh.get('height',1) / pipe_dims_no_obs_gmsh.get('width',3))  if pipe_dims_no_obs_gmsh.get('width',3) > 0 else 4 ))
        dolfin.plot(pipe_mesh_no_obs_gmsh)
        plt.title("Test Pipe - No Obstacle (Gmsh API, direct .msh load)")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("test_pipe_no_obstacle_direct_msh.png")
        plt.close()
        print("Saved test_pipe_no_obstacle_direct_msh.png")
    
    # plt.show() # Comment out for non-interactive runs
    print("\nMesh generation tests complete. Check for .png files.")
