import sys
import json
import makeunitcell_func1 as muc

input_str = sys.argv[1]
input_data = json.loads(input_str)

latticeType = str(input_data['latticeType'])
print(latticeType)
L = int(input_data['size'])
latticeCells = list(map(int, input_data['latticeCells']))
StrutRadius = str(input_data['StrutRadius'])
linp = 'eval(lambda x,y,z: '+ StrutRadius + ')'
StrutRadiusFunc = StrutRadius
StrutRevolutions = str(input_data['StrutRevolutions'])
linp1 = 'eval(lambda x,y,z: '+ StrutRevolutions + ')'
StrutRevFunc = StrutRevolutions
SamplingRate = int(input_data['SamplingRate'])
DoubleNetwork = input_data['DoubleNetwork']
subValues = input_data['subValues']
print(subValues)
subsectionCount = input_data['subsectionCount']
save = input_data['save']
save_STL = input_data['save_STL']
radius = float(input_data['radius'])
print(input_data)

#subsection
#format strutParams function:
finp = '''def strutParams(strutPosition,L):
    x,y,z=float(strutPosition[0]+L),float(strutPosition[1]+L),float(strutPosition[2]+L)
    strutRadius = {inputrad}
    strutRevolutions = {inputrevs}
    {subdivstatement}
    return(float(strutRadius),float(strutRevolutions))'''


subdivs = ''
template = '''if {xmin} <= x <= {xmax} and {ymin} <= y <= {ymax} and {zmin} <= z <= {zmax}: strutRadius, strutRevolutions = {rad}, {revs}'''

for key, values in subValues.items():
    subdiv = template.format(
        xmin=values['subxmin'], xmax=values['subxmax'],
        ymin=values['subymin'], ymax=values['subymax'],
        zmin=values['subzmin'], zmax=values['subzmax'],
        rad=values['subrad'], revs=values['subrevs']
    )
    subdivs += subdiv + '\n    '

finp = finp.format(inputrad=StrutRadius, inputrevs=StrutRevolutions, subdivstatement=subdivs)

print(finp)
exec(finp)

curves = muc.wovenLattice(latticeType, L/2, strutParams, SamplingRate, latticeCells, doubleNetwork=DoubleNetwork)
muc.plotly_plot_curves(curves,L)

if save:
    muc.exportCSV(curves, latticeCells, L/2, doubleNetwork=DoubleNetwork)
    muc.exportOneCSV(curves, latticeCells, L/2)

if save_STL:
    muc.save_piped_stl(curves,radius,12,filename='woven_mesh.stl')
    #curves,rad,pts,lengthtol=rad/2,filename='woven_mesh.stl'