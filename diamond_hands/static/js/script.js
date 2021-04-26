$(function() {
    $('#calendar_home').submit(function(event) {
          event.preventDefault(); // Prevent the form from submitting via the browser

          var spin = $("<span class='spinner-border spinner-border-sm' role='status' aria-hidden='true'></span>");
          $("#submitButtonHome").empty();
          $("#submitButtonHome").append(spin);

          var form = $(this);
          $("#home_graph_results").load(form.attr('action'), form.serialize(), () => {
            $("#submitButtonHome").empty();
            $("#submitButtonHome").text("Submit");
          });
    });

    $('#calendar_graphs').submit(function(event) {
          event.preventDefault(); // Prevent the form from submitting via the browser

          var spin = $("<span class='spinner-border spinner-border-sm' role='status' aria-hidden='true'></span>");
          $("#submitButtonGraph").empty();
          $("#submitButtonGraph").append(spin);

          var form = $(this);
          $("#graph_results").load(form.attr('action'), form.serialize(), () => {
            $("#submitButtonGraph").empty();
            $("#submitButtonGraph").text("Submit");
          });
    });

});