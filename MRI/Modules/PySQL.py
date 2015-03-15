import MySQLdb
from numpy import array, squeeze

class PySQL(object):
    def __init__(self, server="localhost", user="dpc", passwd="dpc", database="NCANDA_RESTING"):
        
        db = MySQLdb.connect(server, user, passwd, database)
        self.cursor = db.cursor()
    
    def get_500_sub_list(self):    
            
            sql = "select sub_id from Subjects500;"
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            
            return squeeze(array(results, str))
    
    def get_all_sub_list(self):    
        
        sql = "select sub_id from AllSubjects;"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        
        return squeeze(array(results, str))
        
    def get_sub_id_output_path(self):
        sql = """SELECT Subjects.sub_id, OutputPaths.output 
                 FROM Subjects 
                 INNER JOIN OutputPaths 
                 ON Subjects.sub_id=OutputPaths.sub_id 
                 ORDER BY Subject.sub_id
                 ;
"""
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        
        return squeeze(array(results, str))
    
    def get_output_path_by_gender(self, gender):
        
        if gender.upper() != 'F' and gender.upper() != 'M':
            raise ValueError, "Must be f/F or m/M."
        
        sql = """SELECT Subjects.sub_id, OutputPaths.output      
                 FROM Subjects      
                 INNER JOIN OutputPaths ON Subjects.sub_id=OutputPaths.sub_id   
                 where Subjects.gender = 'F'   
                 ORDER BY Subjects.sub_id
                 ;
""" %{'gender':gender.upper()}    
 
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        
        return squeeze(array(results, str))