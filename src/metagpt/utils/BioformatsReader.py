"""
This module implements functions to read proprietary images and return their metadata
in OME-XML format and as raw metadata key-value pairs using Bio-Formats.
"""

from typing import Dict, Optional, Union
import javabridge
from bioformats import ImageReader
import javabridge as jutil
from deprecated import deprecated

def get_omexml_metadata(path: Optional[str] = None, url: Optional[str] = None) -> str:
    """
    Read the OME metadata from a file using Bio-Formats.

    Args:
        path (Optional[str]): Path to the file. Defaults to None.
        url (Optional[str]): URL of the file. Defaults to None.

    Returns:
        str: The metadata as XML.

    Raises:
        ValueError: If neither path nor url is provided.
    """
    if not path and not url:
        raise ValueError("Either path or url must be provided")

    with ImageReader(path=path, url=url, perform_init=False) as rdr:
        script = """
        importClass(Packages.loci.common.services.ServiceFactory,
                    Packages.loci.formats.services.OMEXMLService,
                    Packages.loci.formats['in'].DefaultMetadataOptions,
                    Packages.loci.formats['in'].MetadataLevel);
        reader.setGroupFiles(false);
        reader.setOriginalMetadataPopulated(true);
        var service = new ServiceFactory().getInstance(OMEXMLService);
        var metadata = service.createOMEXMLMetadata();
        reader.setMetadataStore(metadata);
        reader.setMetadataOptions(new DefaultMetadataOptions(MetadataLevel.ALL));
        reader.setId(path);
        var xml = service.getOMEXML(metadata);
        xml;
        """
        xml = jutil.run_script(script, dict(path=rdr.path, reader=rdr.rdr))
        return str(xml)

def get_raw_metadata(path: str) -> Dict[str, str]:
    """
    Read the raw metadata from a file using Bio-Formats.

    Args:
        path (str): Path to the file.

    Returns:
        Dict[str, str]: The metadata as a dictionary.
    """
    def get_core_metadata(rdr: ImageReader) -> Dict[str, str]:
        """
        Extract core metadata directly from the reader object.

        Args:
            rdr (ImageReader): The Bio-Formats image reader object.

        Returns:
            Dict[str, str]: Core metadata as a dictionary.
        """
        rdr = rdr.rdr
        core_md = {}
        
        core_methods = [
            'getSizeX', 'getSizeY', 'getSizeZ', 'getSizeC', 'getSizeT',
            'getPixelType', 'getBitsPerPixel', 'getImageCount',
            'getDimensionOrder', 'isRGB', 'isInterleaved',
            'isLittleEndian', 'isOrderCertain', 'isThumbnailSeries',
            'isIndexed', 'isFalseColor', 'getModuloZ', 'getModuloC', 'getModuloT',
            'getThumbSizeX', 'getThumbSizeY', 'getSeriesCount', "getIndex",
        ]
        
        for method in core_methods:
            try:
                value = getattr(rdr, method)()
                core_md[method] = str(value)
            except Exception as e:
                print(f"Error getting {method}: {str(e)}")
        
        return core_md
    
    with ImageReader(path=path, url=None, perform_init=False) as rdr:
        rdr.rdr.setId(path)
        metadata = javabridge.jutil.jdictionary_to_string_dictionary(rdr.rdr.getMetadata(path))
        series_md = javabridge.jutil.jdictionary_to_string_dictionary(rdr.rdr.getSeriesMetadata(path))
        global_md = javabridge.jutil.jdictionary_to_string_dictionary(rdr.rdr.getGlobalMetadata(path))
        core_md = get_core_metadata(rdr)

        meta_all = {**metadata, **series_md, **global_md, **core_md}  # Merges the metadata, potentially overwriting conflicting entries
        return meta_all

def raw_to_tree(raw_metadata: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
    """
    Convert the raw metadata to a tree structure by separating the key on the "|" character.

    Args:
        raw_metadata (Dict[str, str]): The raw metadata dictionary.

    Returns:
        Dict[str, Union[str, Dict]]: The metadata in a tree structure.
    """
    metadata = {}
    for key, value in raw_metadata.items():
        keys = key.split("|")
        current = metadata
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    return metadata