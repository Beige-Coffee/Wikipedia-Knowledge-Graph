let get_text = function() {
    let text_input = $("input#text_input").val()
    return {'text_input': text_input} 

};

let send_text_json = function(input_text) {
    $.ajax({
        url: '/fraud_classifier',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            display_solutions(data);
        },
        data: JSON.stringify(input_text)
    });
};

let display_solutions = function(solutions) {
    $("span#results").html(solutions['prediction'])
};

$(document).ready(function() {

    $("button#classify").click(function() {
        let input_text = get_text();
        send_text_json(input_text);
    })

})