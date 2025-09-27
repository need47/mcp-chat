import json
import sys
from contextlib import asynccontextmanager
from typing import Any, Literal, get_args

import httpx
from mcp.server.fastmcp import Context, FastMCP


@asynccontextmanager
async def http_lifespan(mcp: FastMCP):
    # One shared async client with pooling, keep-alive, HTTP/2, timeouts, and limits.
    client = httpx.AsyncClient(
        http2=True,  # if the server supports it, this improves perf
        timeout=httpx.Timeout(10.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        headers={"User-Agent": "mcp-http-tools/1.0"},
    )
    try:
        yield {
            "http": client,
        }
    finally:
        await client.aclose()


# Initialize FastMCP server
mcp = FastMCP("PUG-REST", dependencies=["httpx"], lifespan=http_lifespan)

# Constants
USER_AGENT = "weather-app/1.0"

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUG_COMPOUND = f"{PUG}/compound"


async def make_request(url: str, client: httpx.AsyncClient) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
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


@mcp.resource("resource://pubchem_compound_property")
async def list_pubchem_compound_property() -> str:
    """List available PubChem compound properties."""
    return "\n".join(list(get_args(PropertyType)))


@mcp.tool()
async def get_pubchem_compound_property(cids: list[int], props: list[PropertyType], ctx: Context) -> str:
    """Get properties for a list of compound IDs (CIDs).

    Args:
        cids (list[int]): List of compound IDs to retrieve properties for
        props (list[PropertyType]): List of properties to retrieve

    Returns:
        str: a JSON array of objects representing compound properties
    """
    if not cids or not props:
        return "No compound IDs or properties specified."

    client: httpx.AsyncClient = ctx.request_context.lifespan_context["http"]

    # Make a request to the PubChem PUG REST API
    url = f"{PUG_COMPOUND}/cid/{','.join(map(str, cids))}/property/{','.join(props)}/JSON"
    data = await make_request(url, client)

    if not data or "PropertyTable" not in data or "Properties" not in data["PropertyTable"]:
        return "Unable to fetch properties from PubChem."

    # Extract and format the properties from the response
    properties = []
    for entry in data["PropertyTable"]["Properties"]:
        cid = entry.get("CID", "Unknown")
        p = {"CID": cid}
        properties.append(p)

        for prop in props:
            values = entry.get(prop)
            if values is None:
                p[prop] = "Not available"
            elif isinstance(values, list):
                p[prop] = ", ".join(map(str, values))
            else:
                p[prop] = values

    return json.dumps(properties, indent=2)


if __name__ == "__main__":
    # Initialize and run the server
    # mcp.run(transport="stdio")
    # Wrap your main server code with exception handling
    try:
        mcp.run(transport="stdio")
    except (BrokenPipeError, ConnectionResetError, EOFError):
        # Client disconnected, exit gracefully
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(0)
