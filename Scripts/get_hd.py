from astroquery.simbad import Simbad

# Set SIMBAD to return HD identifiers
Simbad.add_votable_fields("ids")

# Query by star name
star_name = input("\033[94mEnter star name/identifier: \033[0m")
result = Simbad.query_object(star_name)

# Print all identifiers for the object
if result:
    identifiers = result["ids"][0].split("|")
    hd_id = [id_.strip() for id_ in identifiers if id_.strip().startswith("HD")]
    print(f"HD Identifier(s): \033[92m{hd_id[0]}\033[0m")
else:
    exit("Object not found.")
