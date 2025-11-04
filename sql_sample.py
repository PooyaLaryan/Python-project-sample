from SQL.repo import Repository

r = Repository()

data = r.h3_to_index()
print(data)
print(len(data))
