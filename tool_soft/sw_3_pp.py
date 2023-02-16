from shapely import Polygon, Point, LineString
import geopandas as gp
from matplotlib import pyplot as plt
from IPython import embed


def viz_save(poly, name=''):
    if isinstance(poly, Polygon):
        line = poly.boundary
    else:
        line = poly
    fig = plt.figure()
    plt.plot(*line.xy)
    if not name:
        name = 'foo'
    plt.savefig(f'{name}.png', dpi=300)


def lstr_to_sted(line:LineString):
    ls = []
    for x, y in zip(*line.xy):
        ls.append([x, y])
    return ls


if __name__ == '__main__':

    gdf = gp.read_file('../../data/fpnet008-2/8_0_geojs.geojson')
    print(gdf.head())
    poly = gdf.loc[0]['geometry']
    viz_save(poly)
    linestr = poly.boundary
    lines = lstr_to_sted(linestr)
    for cur_line, next_line in zip(lines[:len(lines)], lines[1:len(lines)]):
        print(cur_line, next_line)
    embed()


