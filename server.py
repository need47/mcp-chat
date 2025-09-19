from typing import Any, Literal, get_args

import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("PUG-REST")

# Constants
USER_AGENT = "weather-app/1.0"

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUG_COMPOUND = f"{PUG}/compound"


async def make_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


PropertyType = Literal[
    "MolecularFormula",
    "MolecularWeight",
    "SMILES",
    "ConnectivitySMILES",
    "InChI",
    "InChIKey",
    "IUPACName",
    "Title",
    "XLogP",
    "ExactMass",
    "MonoisotopicMass",
    "TPSA",
    "Complexity",
    "Charge",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "IsotopeAtomCount",
    "AtomStereoCount",
    "DefinedAtomStereoCount",
    "UndefinedAtomStereoCount",
    "BondStereoCount",
    "DefinedBondStereoCount",
    "UndefinedBondStereoCount",
    "CovalentUnitCount",
    "PatentCount",
    "PatentFamilyCount",
    "AnnotationTypes",
    "AnnotationTypeCount",
    "SourceCategories",
    "LiteratureCount",
    "Volume3D",
    "XStericQuadrupole3D",
    "YStericQuadrupole3D",
    "ZStericQuadrupole3D",
    "FeatureCount3D",
    "FeatureAcceptorCount3D",
    "FeatureDonorCount3D",
    "FeatureAnionCount3D",
    "FeatureCationCount3D",
    "FeatureRingCount3D",
    "FeatureHydrophobeCount3D",
    "ConformerModelRMSD3D",
    "EffectiveRotorCount3D",
    "ConformerCount3D",
    "Fingerprint2D",
]


@mcp.tool()
async def list_pubchem_compound_property() -> str:
    """List available PubChem compound properties."""
    return "\n".join(list(get_args(PropertyType)))


@mcp.tool()
async def get_pubchem_compound_property(cids: list[int], props: list[PropertyType]) -> str:
    """Get properties for a list of compound IDs.

    Args:
        cids (list[int]): List of compound IDs
        props (list[PropertyType]): List of properties to retrieve
    """
    if not cids or not props:
        return "No compound IDs or properties specified."

    # Make a request to the PubChem PUG REST API
    url = f"{PUG_COMPOUND}/cid/{','.join(map(str, cids))}/property/{','.join(props)}/JSON"
    data = await make_request(url)

    if not data or "PropertyTable" not in data or "Properties" not in data["PropertyTable"]:
        return "Unable to fetch properties from PubChem."

    # Extract and format the properties from the response
    properties = []
    for entry in data["PropertyTable"]["Properties"]:
        cid = entry.get("CID", "Unknown")
        for prop in props:
            values = entry.get(prop)
            if values is None:
                properties.append(f"CID {cid} - {prop}: Not available")
            elif isinstance(values, list):
                properties.append(f"CID {cid} - {prop}: {', '.join(map(str, values))}")
            else:
                properties.append(f"CID {cid} - {prop}: {values}")

    return "\n---\n".join(properties)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
