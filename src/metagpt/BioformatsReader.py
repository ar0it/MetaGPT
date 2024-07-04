"""
This file implements functions to read the proprietary images and returns their metadata in OME-XML format and the
raw metadata key-value pairs.
"""
import javabridge
from bioformats import ImageReader
import javabridge as jutil
import numpy as np
from deprecated import deprecated


def get_omexml_metadata(path=None, url=None):
    '''Read the OME metadata from a file using Bio-formats

    :param path: path to the file

    :param groupfiles: utilize the groupfiles option to take the directory structure
                 into account.

    :returns: the metdata as XML.

    '''
    with ImageReader(path=path, url=url, perform_init=False) as rdr:
        #
        # Below, "in" is a keyword and Rhino's parser is just a little wonky I fear.
        #
        # It is critical that setGroupFiles be set to false, goodness knows
        # why, but if you don't the series count is wrong for flex files.
        #
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
        #print(type(xml))
        return str(xml)



def get_raw_metadata(path: str = None) -> dict[str, str]:
    """
    Read the raw metadata from a file using Bio-formats

    :param path: path to the file
    :return: the metadata as a dictionary
    """
    def get_core_metadata(rdr) -> dict[str, str]:
        """
        Extract core metadata directly from the rdr object
        """
        core_md = {}
        
        # List of core metadata methods to call
        core_methods = [
            'getSizeX', 'getSizeY', 'getSizeZ', 'getSizeC', 'getSizeT',
            'getPixelType', 'getBitsPerPixel', 'getImageCount',
            'getDimensionOrder', 'isRGB', 'isInterleaved',
            'isLittleEndian', 'isOrderCertain', 'isThumbnailSeries',
            'isIndexed', 'isFalseColor', 'getModuloZ', 'getModuloC', 'getModuloT',
            'getThumbSizeX', 'getThumbSizeY', 'getSeriesCount',
        ]
        
        # Call each method and store the result
        for method in core_methods:
            try:
                value = getattr(rdr, method)()
                core_md[method] = str(value)
            except Exception as e:
                print(f"Error getting {method}: {str(e)}")
        """
        # Get metadata for each series
        series_count = rdr.getSeriesCount()
        for series in range(series_count):
            rdr.setSeries(series)
            for method in core_methods:
                try:
                    value = getattr(rdr, method)()
                    core_md[f"Series_{series}_{method}"] = str(value)
                except Exception as e:
                    print(f"Error getting {method} for series {series}: {str(e)}")
        """
        
        return core_md
    
    with ImageReader(path=path, url=None, perform_init=False) as rdr:
        rdr.rdr.get
        rdr.rdr.setId(path)
        metadata  = javabridge.jutil.jdictionary_to_string_dictionary(rdr.rdr.getMetadata(path))
        series_md = javabridge.jutil.jdictionary_to_string_dictionary(rdr.rdr.getSeriesMetadata(path))
        global_md = javabridge.jutil.jdictionary_to_string_dictionary(rdr.rdr.getGlobalMetadata(path))
        core_md = get_core_metadata(rdr.rdr)

        meta_all = metadata | series_md | global_md | core_md # merges the metadata overwrite potentially conflicting entries
        return meta_all

def raw_to_tree(raw_metadata: dict[str, str]):
    """
    Convert the raw metadata to a tree structure, by seperating the key on the "|" character.
    """
    metadata = {}
    for key, value in raw_metadata.items():
        keys = key.split("|")
        current = metadata
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    # pretty print dictionary
    import json
    #print(json.dumps(metadata, indent=4))
    return metadata