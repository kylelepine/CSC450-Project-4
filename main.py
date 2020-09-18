import psycopg2 


conn = None


def connect():
    global conn
    # conn parameters
    dbname = 'CSC-450_FDS'
    pword = 'Apcid28;6jdn'
    try:
        # Attempts to connect to server
        print("Connecting...")
        conn = psycopg2.connect(host = 'localhost', database = dbname, user = 'postgres', password = pword)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    
    print("Connection successful.")
    print_db_version()       


def disconnect():
    # closes communcation with PostgreSQL server
    global conn
    print("Disconnecting...")
    if conn is not None:
        conn.close()
    print("Disconnection successful.")


def print_db_version():
    print('PostgreSQL database version:')
    try:
        curr = conn.cursor()
        curr.execute('SELECT version()')
        db_version = curr.fetchone()
        curr.close()
        print(db_version)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def manage_database_menu():
    print("manage_database_menu() called")
    show_manage_menu = True
    while show_manage_menu:
        print("""
        Command:        Description:\n
        1         ---   Add template.\n
        2         ---   Delete template.\n
        3         ---   Modify template.\n
        r         ---   Return to previous menu.
        """)
        command = input("Enter a command: ")
        if command == "1":
            print(f"{command} selected.")
            image_path = input("Please enter the path of the image: ")
            verify_image_path(image_path)
            template_type = input("Please enter the template_type: ")
            add_template(image_path, template_type)
        elif command == "2":
            print(f"{command} selected.")
            delete_template_id = input("Enter the template_id you wish to delete: ")
            delete_template(delete_template_id)
        elif command == "3":
            # TODO: Add functionality to modify template
            print(f"{command} selected.")
        elif command == "r":
            show_manage_menu = False
        else:
            print(f"{command} is not a valid command.\nPlease try again.")


def verify_image_path(image_path):
    print("Image verified.")

def add_template(image_path, template_type):
    #TODO: add functionality to extract image name from path.
    #      Maybe find last '/' and have name = what's left.
    global conn
    image_name = image_path
    try:
        curr = conn.cursor()
        query = f"""
        INSERT INTO template (template_type, image_name, image) VALUES
        ('{template_type}', '{image_name}', bytea('{image_path}'));
         """
        curr.execute(query)
        conn.commit()
        curr.close()
        print("Added template successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    curr.close()
    

def delete_template(template_id):
    global conn
    try:
        curr = conn.cursor()
        query = f"""
        DELETE FROM template
        WHERE template_id = {template_id}
        """
        curr.execute(query)
        conn.commit()
        curr.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def view_template_menu():
    print("view_template_menu() called")
    show_view_template_menu = True
    while show_view_template_menu:
        print("""
        Command:        Description:\n
        1         ---   View image by template_id.\n
        r         ---   Return to previous menu.
        """)
        command = input("Enter a command: ")
        if command == '1':
            print(f"{command} selected.")
            template_id = input("Enter template_id: ")
            look_up_image_by_template_id(template_id)
        elif command =='r':
            show_view_template_menu = False
        else:
            print(f"{command} is not a valid command.\nPlease try again.")

def look_up_image_by_template_id(template_id):
    global conn
    try:
        curr = conn.cursor()
        query = f"""
        SELECT image
        FROM template
        WHERE template_id = {template_id}
        """
        curr.execute(query)
        image = curr.fetchone()
        print(image)
        image = bytes(image)
        print(image)
        curr.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def quit_application():
    print("quit_application() called")
    disconnect()


def main():
    connect()
    application_running = True
    
    while application_running:
        print("""
        Command:       Description:\n
        Manage(m)    --   Add, Delete, or Modify information stored in Database.\n
        View(v)      --   View database entries.\n
        Quit(q)      --   Terminates program and disconnects from postgreSQL server.
        """)

        user_command = input("Enter a command: ")
        if user_command == "v":
            view_template_menu()
        elif user_command == "m":
            manage_database_menu()
        elif user_command == "q":
            application_running = False
        else:
            print("\nINCORRECT COMMAND ENTERED")
    
    quit_application()


if __name__ == '__main__':
    main()
    print("FINISHED")