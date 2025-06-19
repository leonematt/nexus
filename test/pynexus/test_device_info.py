import nexus

info = nexus.lookup_chip_info('apple-gpu-applegpu_g16s')


def checkInfoInt(path, defval):
  lval = info.get_int(path)
  print(f"Check: {lval} == {defval}")
  assert(lval == defval)

def checkInfoStr(path, defval):
  lval = info.get_str(path)
  assert(lval == defval)


checkInfoStr('Name', 'Apple GPU G16S')
checkInfoStr('Architecture', 'applegpu_g16s')
checkInfoInt('ReleaseYear', 2024)

