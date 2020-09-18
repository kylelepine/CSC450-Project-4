CREATE TABLE Template (
	Template_ID serial PRIMARY KEY,
	template_type varchar(10) NOT NULL CONSTRAINT valid_template_type CHECK (template_type IN ('standing', 'falling', 'sitting', 'lying')),
	image_name varchar(50),
	image bytea NOT NULL
);