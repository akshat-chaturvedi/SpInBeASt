# from SpectrumAnalyzer import spin_beast

def coord_transfer(ra: str, dec: str):
    """
    Converts the RA and Declination of an astronomical source from sexagesimal strings to decimal degrees

    :param ra: (str) Right Ascension (in sexagesimal) of object

    :param dec: (str) Declination (in sexagesimal) of object
    """
    ra_string = ra.split(" ")
    ra_hh = int(ra_string[0])
    ra_mm = int(ra_string[1])
    ra_ss = float(ra_string[2])

    dec_string = dec.split(" ")
    dec_deg = int(dec_string[0])
    dec_min = int(dec_string[1])
    dec_sec = float(dec_string[2])

    ra_deg = ra_hh * 15 + ra_mm / 60 * 15 + ra_ss / 3600 * 15
    if dec_deg < 0:
        dec_deg = dec_deg - dec_min / 60 - dec_sec / 3600
    else:
        dec_deg = dec_deg + dec_min / 60 + dec_sec / 3600

    return ra_deg, dec_deg


if __name__ == "__main__":
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    print(spin_beast)
    # print(f"""
    #             ########################## {RED}COO{RESET}{GREEN}RDIN{RESET}{YELLOW}ATE{RESET}{BLUE} TRANSFORMER{RESET} ##########################
    #         """)
    RA = input("Enter RA: ")
    DEC = input("Enter Dec: ")
    ra_conv, dec_conv = coord_transfer(RA, DEC)
    print(f"Converted RA: {ra_conv:.4f} deg")
    print(f"Converted Dec: {dec_conv:.4f} deg")
