from astropy.time import Time

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# (M)JD to Date Converter

print(f"""
            ########################## {RED}(M)JD{RESET}{GREEN} TO{RESET}{YELLOW} DATE{RESET}{BLUE} CONVERTER{RESET} ##########################                          
                           https://github.com/akshat-chaturvedi/SpInBeASt                
            ############################################################################
        """)

jd_or_mjd_check = input(
    "Do you want to convert just a JD date [enter], or an MJD date [any other key]?: "
)

if jd_or_mjd_check == "":
    date = input("Enter the JD date: ")
    fits = Time(date, format="jd", scale="utc").fits

else:
    date = input("Enter the MJD date: ")
    fits = Time(date, format="mjd", scale="utc").fits

print(f"Converted Date: {GREEN}{fits.split('T')[0]}{RESET}")
