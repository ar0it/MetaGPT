from ome_types import OME

def from_dict(ome_dict):
    """
    Convert a dictionary to an OME object.
    """
    def set_attributes(obj, data):
        for key, value in data.items():
            if isinstance(value, dict):
                attr = getattr(obj, key, None)
                if attr is not None:
                    set_attributes(attr, value)
                else:
                    setattr(obj, key, value)
            elif isinstance(value, list):
                # Assume the list items are dictionaries that need similar treatment
                existing_list = getattr(obj, key, [])
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        if i < len(existing_list):
                            set_attributes(existing_list[i], item)
                        else:
                            # Handle the case where the list item does not already exist
                            existing_list.append(item)
                    else:
                        existing_list.append(item)
                setattr(obj, key, existing_list)
            else:
                setattr(obj, key, value)
    
    ome = OME()
    set_attributes(ome, ome_dict)
    return ome

def _init_logger():
    """This is so that Javabridge doesn't spill out a lot of DEBUG messages
    during runtime.
    From CellProfiler/python-bioformats.
    """
    import javabridge as jb
    rootLoggerName = jb.get_static_field("org/slf4j/Logger",
                                         "ROOT_LOGGER_NAME",
                                         "Ljava/lang/String;")

    rootLogger = jb.static_call("org/slf4j/LoggerFactory",
                                "getLogger",
                                "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                rootLoggerName)

    logLevel = jb.get_static_field("ch/qos/logback/classic/Level",
                                   "ERROR",
                                   "Lch/qos/logback/classic/Level;")

    jb.call(rootLogger,
            "setLevel",
            "(Lch/qos/logback/classic/Level;)V",
            logLevel)