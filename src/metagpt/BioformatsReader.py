"""
This file implements functions to read the proprietary images and returns their metadata in OME-XML format and the
raw metadata key-value pairs.
"""
import javabridge
import imagej
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
        print(type(xml))
        return str(xml)


@deprecated()
def get_raw_metadata2(path=None, url=None):
    with ImageReader(path=path, url=url, perform_init=False) as rdr:
        # Below, "in" is a keyword and Rhino's parser is just a little wonky I fear.
        #
        # It is critical that setGroupFiles be set to false, goodness knows
        # why, but if you don't the series count is wrong for flex files.
        #
        script = """
            importClass(Packages.loci.common.services.ServiceFactory,
                        Packages.loci.formats.services.OMEXMLService,
                        Packages.loci.formats['in'].DefaultMetadataOptions,
                        Packages.loci.formats['in'].MetadataLevel,
                        java.util.Hashtable);
            reader.setGroupFiles(false);
            reader.setOriginalMetadataPopulated(true);
            var service = new ServiceFactory().getInstance(OMEXMLService);
            var metadata = service.createOMEXMLMetadata();
            reader.setMetadataStore(metadata);
            reader.setMetadataOptions(new DefaultMetadataOptions(MetadataLevel.ALL));
            reader.setId(path);
            var globalMetadata = reader.getGlobalMetadata();
            var metadataDict = {};
            var keys = globalMetadata.keys();
            while (keys.hasMoreElements()) {
                var key = keys.nextElement();
                metadataDict[key] = globalMetadata.get(key);
            }
            metadataDict;
            """
        metadata_dict = jutil.run_script(script, dict(path=rdr.path, reader=rdr.rdr))

    def java_hashtable_to_dict(java_hashtable):
        """Convert a Java Hashtable to a Python dictionary."""
        dict = {}
        keys = javabridge.jutil.to_string(java_hashtable)
        for key in javabridge.jutil.iterate(java_hashtable.keys()):
            dict[key] = javabridge.jutil.to_string(java_hashtable.get(key))
        return dict

    metadata_dict = java_hashtable_to_dict(metadata_dict)
    return metadata_dict


def get_raw_metadata(path: str = None):
    """
    Read the raw metadata from a file using Bio-formats

    python flavored macro recording in fiji shows this:
    IJ.run("Bio-Formats Importer", "open=/home/aaron/Documents/Projects/MetaGPT/in/images/Image_8.czi autoscale color_mode=Default display_metadata rois_import=[ROI manager] view=[Metadata only] stack_order=Default");
    IJ.saveAs("Text", "/home/aaron/Desktop/Original Metadata - Image_8.csv");

    macro flavor:

    """
    with ImageReader(path=path, url=None, perform_init=False) as rdr:
        rdr.rdr.setId(path)
        series_md = javabridge.jutil.jdictionary_to_string_dictionary(rdr.rdr.getSeriesMetadata(path))
        global_md = javabridge.jutil.jdictionary_to_string_dictionary(rdr.rdr.getGlobalMetadata(path))
        return global_md


import imagej
import os
import pandas as pd

def get_raw_metadata(image_path):
    # Initialize ImageJ
    ij = imagej.init('sc.fiji:fiji')

    # Construct the Bio-Formats Importer command
    bio_formats_command = f"open={image_path} autoscale color_mode=Default display_metadata rois_import=[ROI manager] view=[Metadata only] stack_order=Default"
    
    # Run the Bio-Formats Importer
    ij.py.run_macro(f"IJ.run('Bio-Formats Importer', '{bio_formats_command}');")

    # Get the metadata
    metadata = ij.WindowManager.getCurrentImage().getStringProperty('Info')

    # Split metadata into lines and then key-value pairs
    metadata_lines = metadata.split('\n')
    metadata_dict = {}
    for line in metadata_lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            metadata_dict[key] = value

    print("Metadata:", metadata_dict)

# Example usage
image_path = '/home/aaron/Documents/Projects/MetaGPT/in/images/Image_8.czi'
get_raw_metadata(image_path)
