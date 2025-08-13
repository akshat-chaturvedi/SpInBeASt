from astropy.time import Time

date_or_date_and_time_check = input(
    "\033[94mDo you want to convert just a date [enter], or a date with a time [any other key]?: \033[0m"
)

if date_or_date_and_time_check == "":
    date = input(
        "\033[94mEnter the date you want to convert in YYYY-MM-DD format: \033[0m"
    )

else:
    date = input(
        "\033[94mEnter the date you want to convert in YYYY-MM-DDTHH:MM:SS format: \033[0m"
    )


jd = Time(date, format="isot", scale="utc").jd

print(f"\033[94mConverted Date:\033[0m \033[92m{jd:.4f} days\033[0m")
