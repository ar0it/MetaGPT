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