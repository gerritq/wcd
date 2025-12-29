import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D

# resource availability
LANGS = {
    "high": ["en", "pt", "de", "ru", "it", "vi", "tr", "nl"],
    "medium": ["uk", "ro", "id", "bg", "uz"],
    "low": ["no", "az", "mk", "hy", "sq"],
}

# language families + macro families
LANG_FAMILY = {
    "germanic": {
        "macro": "Indo-European",
        "langs": ["en", "de", "nl", "no"],
    },
    "romance": {
        "macro": "Indo-European",
        "langs": ["pt", "it", "ro", "sq"],
    },
    "slavic": {
        "macro": "Indo-European",
        "langs": ["ru", "uk", "bg", "mk"],
    },
    "turkic": {
        "macro": "Turkic",
        "langs": ["tr", "az", "uz"],
    },
    "austronesian": {
        "macro": "Austronesian",
        "langs": ["id"],
    },
    "vietic": {
        "macro": "Austroasiatic",
        "langs": ["vi"],
    },
    "armenian": {
        "macro": "Indo-European",
        "langs": ["hy"],
    },
}

# lang -> (lat, lon)
CAPITALS = {
    "en": (51.5074, -0.1278),
    "pt": (38.7223, -9.1393),
    "de": (52.5200, 13.4050),
    "ru": (55.7558, 37.6173),
    "it": (41.9028, 12.4964),
    "vi": (21.0285, 105.8542),
    "tr": (39.9334, 32.8597),
    "nl": (52.3676, 4.9041),
    "uk": (50.4501, 30.5234),
    "ro": (44.4268, 26.1025),
    "id": (-6.2088, 106.8456),
    "bg": (42.6977, 23.3219),
    "uz": (41.2995, 69.2401),
    "no": (59.9139, 10.7522),
    "az": (40.4093, 49.8671),
    "mk": (41.9981, 21.4254),
    "hy": (40.1872, 44.5152),
    "sq": (41.3275, 19.8187),
}

# colors = language families
FAMILY_COLORS = {
    "germanic": "tab:blue",
    "romance": "tab:orange",
    "slavic": "tab:green",
    "turkic": "tab:red",
    "austronesian": "tab:purple",
    "vietic": "tab:brown",
    "armenian": "tab:pink",
}

# markers = resource availability
RESOURCE_MARKERS = {
    "high": "o",
    "medium": "s",
    "low": "^",
}

def get_family(lang):
    for fam, info in LANG_FAMILY.items():
        if lang in info["langs"]:
            return fam
    return None

# --- figure ---
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([-15, 115, -10, 65], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black", linewidth=0.4)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4)

gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                  color="gray", alpha=0.6, linestyle="--")
gl.bottom_labels = False
gl.top_labels = True
gl.left_labels = True
gl.right_labels = False
gl.xlabel_style = {"size": 9}
gl.ylabel_style = {"size": 9}

# plot points (filled, unchanged)
for resource, langs in LANGS.items():
    for lang in langs:
        lat, lon = CAPITALS[lang]
        fam = get_family(lang)

        ax.plot(
            lon, lat,
            marker=RESOURCE_MARKERS[resource],
            color=FAMILY_COLORS[fam],
            markersize=7,
            linestyle="None",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

# --- legend 1: language families (outlined only) — bottom left
family_legend = []
for fam, info in LANG_FAMILY.items():
    label = f"{info['macro']} ({fam.capitalize()})"
    family_legend.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=FAMILY_COLORS[fam],  # solid color
            markeredgecolor="none",              # no outline
            label=label,
            markersize=9,
        )
    )

leg1 = ax.legend(
    handles=family_legend,
    loc="lower left",
    frameon=True,
    fontsize=11,
    title="Language family",
    title_fontsize=13,
)
ax.add_artist(leg1)

# --- legend 2: resource availability (outlined only) — top right
resource_legend = [
    Line2D(
        [0], [0],
        marker=RESOURCE_MARKERS[r],
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1,
        label=f"{r.capitalize()} resource",
        markersize=9,
    )
    for r in RESOURCE_MARKERS
]

ax.legend(
    handles=resource_legend,
    loc="upper right",
    frameon=True,
    fontsize=11,
    title="Resource level",
    title_fontsize=13,
)

plt.tight_layout()
plt.savefig("plots/world_map.pdf", dpi=300)
# plt.show()