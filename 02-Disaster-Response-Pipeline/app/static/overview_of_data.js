var genre = ['news','direct','social']
var count = [13068,10782,2398]

var trace1 = {
	x:genre,
	y:count,
	type: 'bar'
};

var data = [trace1];

var layout = {
	
	title:"Distribution of Message Genres",
	xaxis:{title:"Genre"},
	yaxis:{title:"Count"}
}

Plotly.newPlot("overview_of_data",data,layout);