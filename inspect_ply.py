#!/usr/bin/env python3
"""
PLY File Inspector - Read and understand what's inside a PLY file.

PLY (Polygon File Format) is a simple text-based 3D geometry format.
It stores geometry, colors, normals, and other properties for 3D points/meshes.

Usage:
    python inspect_ply.py <path_to_file.ply>
"""

import numpy as np
from pathlib import Path
from plyfile import PlyData


def inspect_ply(ply_path):
    """
    Read a PLY file and print its contents in detail.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        print(f"❌ Error: File not found: {ply_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"PLY FILE INSPECTION: {ply_path.name}")
    print(f"{'='*80}\n")
    
    # Read PLY file
    ply = PlyData.read(ply_path)
    
    # Print file format info
    print(f"📋 FILE FORMAT:")
    print(f"  Format: {ply.text if ply.text else 'Binary'}")
    print(f"  Version: {ply.comments}\n")
    
    # List all elements in the file
    print(f"📊 ELEMENTS ({len(ply.elements)} total):")
    print(f"  {', '.join([e.name for e in ply.elements])}\n")
    
    # Detailed inspection of each element
    for element in ply.elements:
        print(f"{'-'*80}")
        print(f"📦 ELEMENT: {element.name.upper()}")
        print(f"{'─'*80}")
        print(f"  Count: {element.count} items")
        print(f"  Properties: {len(element.properties)} fields\n")
        
        print(f"  Properties:")
        for prop in element.properties:
            print(f"    • {prop.name:<20} ({prop.val_dtype})")
        
        # Try to print first few rows
        if element.count > 0:
            print(f"\n  First 3 {element.name}:")
            
            for i in range(min(3, element.count)):
                row_data = []
                for prop in element.properties:
                    val = element.data[prop.name][i]
                    # Format value nicely
                    if isinstance(val, (np.floating, float)):
                        row_data.append(f"{val:.4f}")
                    else:
                        row_data.append(str(val))
                
                print(f"    [{i}]: {', '.join(row_data)}")
        
        print()
    
    # Print summary statistics
    if 'vertex' in {e.name for e in ply.elements}:
        vertices = ply['vertex']
        print(f"{'-'*80}")
        print(f"📈 STATISTICS:")
        print(f"{'─'*80}\n")
        
        if 'x' in vertices.data.dtype.names:
            x = vertices.data['x']
            y = vertices.data['y']
            z = vertices.data['z']
            
            print(f"  XYZ Coordinates:")
            print(f"    X: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
            print(f"    Y: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
            print(f"    Z: min={z.min():.4f}, max={z.max():.4f}, mean={z.mean():.4f}\n")
        
        if 'red' in vertices.data.dtype.names:
            r = vertices.data['red']
            g = vertices.data['green']
            b = vertices.data['blue']
            print(f"  RGB Colors:")
            print(f"    R: min={r.min()}, max={r.max()}, mean={r.mean():.1f}")
            print(f"    G: min={g.min()}, max={g.max()}, mean={g.mean():.1f}")
            print(f"    B: min={b.min()}, max={b.max()}, mean={b.mean():.1f}\n")
        
        if 'nx' in vertices.data.dtype.names:
            nx = vertices.data['nx']
            ny = vertices.data['ny']
            nz = vertices.data['nz']
            print(f"  Normals (XYZ):")
            print(f"    NX: min={nx.min():.4f}, max={nx.max():.4f}")
            print(f"    NY: min={ny.min():.4f}, max={ny.max():.4f}")
            print(f"    NZ: min={nz.min():.4f}, max={nz.max():.4f}\n")
        
        if 'alpha' in vertices.data.dtype.names:
            alpha = vertices.data['alpha']
            print(f"  Alpha (Opacity):")
            print(f"    min={alpha.min()}, max={alpha.max()}, mean={alpha.mean():.2f}\n")
    
    print(f"{'='*80}\n")


def show_ply_structure():
    """Show what a typical PLY file structure looks like."""
    print("""
PLY FILE STRUCTURE (Example - first 30 lines):
════════════════════════════════════════════════════════════════════════

ply                                  ← Magic header
format ascii 1.0                     ← Format (ascii or binary)
comment Created by Gaussian Splatting
comment ...
element vertex 123456               ← Element definition: 123456 vertices
property float x                     ← Property: X coordinate
property float y                     ← Property: Y coordinate
property float z                     ← Property: Z coordinate
property float nx                    ← Property: Normal X
property float ny                    ← Property: Normal Y
property float nz                    ← Property: Normal Z
property uchar red                   ← Property: Red color (0-255)
property uchar green                 ← Property: Green color (0-255)
property uchar blue                  ← Property: Blue color (0-255)
property uchar alpha                 ← Property: Opacity/Transparency
end_header                           ← End of header, data follows

0.123 -0.456 1.234 0.0 0.0 1.0 255 128 64 255    ← Vertex 1 data
0.124 -0.457 1.235 0.0 0.0 1.0 255 128 64 255    ← Vertex 2 data
...
════════════════════════════════════════════════════════════════════════

COMMON ELEMENTS:
  • vertex: 3D points with properties (xyz, normals, colors, etc.)
  • face: Triangle/polygon connectivity (vertex indices)

COMMON PROPERTIES:
  • x, y, z: 3D coordinates (float)
  • nx, ny, nz: Surface normals (float)
  • red, green, blue: Color components (uchar 0-255)
  • alpha: Transparency (uchar 0-255)
  • confidence: How confident the point is (float 0-1)
  • intensity: Light intensity (float/int)
    """)


def read_ply_raw(ply_path):
    """
    Read PLY file and return raw data for manual inspection.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        print(f"❌ Error: File not found: {ply_path}")
        return None
    
    ply = PlyData.read(ply_path)
    return ply


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inspect_ply.py <ply_file>")
        print("\nExamples:")
        print("  python inspect_ply.py input.ply")
        print("  python inspect_ply.py point_cloud.ply")
        print("\nTo understand PLY format, see the function docstrings above.")
        show_ply_structure()
    else:
        inspect_ply(sys.argv[1])
