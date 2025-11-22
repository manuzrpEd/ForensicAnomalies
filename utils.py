DOMAIN = "data.lacity.org"

OFFENSES_ID = "y8y3-fqfu"
VICTIMS_ID = "gqf2-vm2j"

DESC_VARIABLES = [
    "area_name",
    "totaloffensecount",
    "group",
    "nibr_description",
    "crime_against",
    "premis_desc",
    "status_desc",
    'totalvictimcount', 'victim_shot', 'domestic_violence_crime',
    'hate_crime', 'gang_related_crime', 'transit_related_crime',
    'homeless_victim_crime', 'homeless_suspect_crime',
    'homeless_arrestee_crime', 'weapon_desc',
    'vict_age',
    'vict_descent', 'vict_sex', 'victim_type'
]

SCHEMA = {
"offenses": {
    "caseno": "string",
    "uniquenibrno": "string",
    "date_rptd": "datetime",
    "date_occ": "datetime",
    "time_occ": "int",                   # typically HHMM format
    "area": "int",
    "area_name": "string",
    "rpt_dist_no": "string",
    "totaloffensecount": "int",
    "group": "string",
    "nibr_code": "string",
    "nibr_description": "string",
    "crime_against": "string",
    "premis_cd": "string",
    "premis_desc": "string",
    "status": "string",
    "status_desc": "string",
    "totalvictimcount": "int",
    "victim_shot": "str",               # Yes/No
    "domestic_violence_crime": "str",
    "hate_crime": "str",
    "gang_related_crime": "str",
    "transit_related_crime": "str",
    "homeless_victim_crime": "str",
    "homeless_suspect_crime": "str",
    "homeless_arrestee_crime": "str",
    "weapon_used_cd": "string",
    "weapon_desc": "string"
    },
"victims": {
    "caseno": "string",
    "uniquevictimno": "string",
    "date_rptd": "datetime",
    "date_occ": "datetime",
    "time_occ": "int",
    "area": "int",
    "area_name": "string",
    "rpt_dist_no": "string",
    "totalvictimcount": "int",
    "victim_type": "string",
    "victim_shot": "str",           # Yes/No
    "status": "string",
    "status_desc": "string",
    "vict_age": "int",
    "vict_sex": "string",
    "vict_descent": "string"
    }
}